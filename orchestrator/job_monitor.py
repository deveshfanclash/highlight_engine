"""
Job Monitor

Monitors AWS Batch jobs and provides:
- Health check based on heartbeats
- Automatic restart of failed jobs
- Stuck job detection
- Alerting via SNS

Designed to run as:
1. Lambda function triggered by EventBridge (every 1-5 minutes)
2. Background thread in a long-running process
3. CLI tool for manual checks

Usage:
    # As Lambda
    from orchestrator.job_monitor import monitor_handler
    # Configure EventBridge to invoke every minute

    # As CLI
    python -m orchestrator.job_monitor --match-id match_123

    # Programmatically
    from orchestrator.job_monitor import JobMonitor
    monitor = JobMonitor()
    result = monitor.check_and_restart(match_id="match_123")
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MonitorConfig:
    """Configuration for job monitoring"""
    # Thresholds
    heartbeat_timeout_seconds: int = 300  # 5 minutes without heartbeat = stuck
    max_retries: int = 3
    job_timeout_minutes: int = 180  # 3 hours max per job

    # AWS resources
    dynamodb_table: str = "inference_jobs"
    sns_topic_arn: Optional[str] = None  # For alerts

    # Batch queues (for restart)
    gpu_job_queue: str = "inference-gpu-queue"
    cpu_job_queue: str = "inference-cpu-queue"

    @classmethod
    def from_env(cls) -> "MonitorConfig":
        return cls(
            heartbeat_timeout_seconds=int(os.getenv("HEARTBEAT_TIMEOUT_SECONDS", "300")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            job_timeout_minutes=int(os.getenv("JOB_TIMEOUT_MINUTES", "180")),
            dynamodb_table=os.getenv("DYNAMODB_TABLE", "inference_jobs"),
            sns_topic_arn=os.getenv("SNS_ALERT_TOPIC"),
            gpu_job_queue=os.getenv("BATCH_GPU_QUEUE", "inference-gpu-queue"),
            cpu_job_queue=os.getenv("BATCH_CPU_QUEUE", "inference-cpu-queue"),
        )


class JobHealth(str, Enum):
    """Health status of a job"""
    HEALTHY = "HEALTHY"
    STUCK = "STUCK"  # No heartbeat
    FAILED = "FAILED"  # AWS Batch reported failure
    TIMEOUT = "TIMEOUT"  # Running too long
    UNKNOWN = "UNKNOWN"


@dataclass
class JobHealthReport:
    """Health report for a single job"""
    service_id: str
    job_id: str
    health: JobHealth
    batch_status: str
    last_heartbeat: Optional[datetime]
    seconds_since_heartbeat: Optional[int]
    retry_count: int
    error_message: Optional[str]
    action_taken: Optional[str] = None


@dataclass
class MatchHealthReport:
    """Health report for an entire match"""
    match_id: str
    overall_health: JobHealth
    job_count: int
    healthy_count: int
    failed_count: int
    stuck_count: int
    completed_count: int
    jobs: List[JobHealthReport]
    actions_taken: List[str]


# =============================================================================
# JOB MONITOR
# =============================================================================

class JobMonitor:
    """
    Monitors and manages AWS Batch jobs.

    Responsibilities:
    - Query job status from AWS Batch
    - Check heartbeats in DynamoDB
    - Detect failed/stuck jobs
    - Restart jobs (up to max_retries)
    - Send alerts for persistent failures
    """

    def __init__(self, config: Optional[MonitorConfig] = None):
        self.config = config or MonitorConfig.from_env()
        self._batch_client = None
        self._dynamodb = None
        self._sns_client = None

    @property
    def batch_client(self):
        if self._batch_client is None:
            import boto3
            self._batch_client = boto3.client(
                "batch",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        return self._batch_client

    @property
    def dynamodb(self):
        if self._dynamodb is None:
            import boto3
            self._dynamodb = boto3.resource(
                "dynamodb",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        return self._dynamodb

    @property
    def sns_client(self):
        if self._sns_client is None and self.config.sns_topic_arn:
            import boto3
            self._sns_client = boto3.client(
                "sns",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
        return self._sns_client

    def _get_batch_job_status(self, job_ids: List[str]) -> Dict[str, Dict]:
        """Get status of multiple jobs from AWS Batch"""
        if not job_ids:
            return {}

        # AWS Batch allows max 100 jobs per describe call
        results = {}
        for i in range(0, len(job_ids), 100):
            batch = job_ids[i:i+100]
            response = self.batch_client.describe_jobs(jobs=batch)

            for job in response.get("jobs", []):
                results[job["jobId"]] = {
                    "status": job["status"],
                    "statusReason": job.get("statusReason"),
                    "createdAt": job.get("createdAt"),
                    "startedAt": job.get("startedAt"),
                    "stoppedAt": job.get("stoppedAt"),
                    "container": job.get("container", {}),
                }

        return results

    def _get_jobs_for_match(self, match_id: str) -> List[Dict]:
        """Get all jobs for a match from DynamoDB"""
        table = self.dynamodb.Table(self.config.dynamodb_table)

        response = table.query(
            KeyConditionExpression="pk = :pk",
            ExpressionAttributeValues={
                ":pk": f"{match_id}#jobs"
            }
        )

        return response.get("Items", [])

    def _get_service_status(self, match_id: str, service_id: str) -> Optional[Dict]:
        """Get service status from inference_results table (heartbeat source)"""
        try:
            # Services write status to inference_results table
            import boto3
            dynamodb = boto3.resource(
                "dynamodb",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            table = dynamodb.Table(os.getenv("INFERENCE_RESULTS_TABLE", "inference_results"))

            response = table.get_item(
                Key={
                    "pk": f"{match_id}#service_status",
                    "sk": service_id
                }
            )
            return response.get("Item")

        except Exception as e:
            logger.warning(f"Failed to get service status: {e}")
            return None

    def _check_job_health(
        self,
        job_record: Dict,
        batch_status: Dict,
        service_status: Optional[Dict],
        now: datetime
    ) -> JobHealthReport:
        """Determine health of a single job"""
        service_id = job_record["service_id"]
        job_id = job_record["job_id"]
        retry_count = job_record.get("retry_count", 0)

        # Get batch status
        status = batch_status.get("status", "UNKNOWN") if batch_status else "UNKNOWN"

        # Check for completed
        if status == "SUCCEEDED":
            return JobHealthReport(
                service_id=service_id,
                job_id=job_id,
                health=JobHealth.HEALTHY,
                batch_status=status,
                last_heartbeat=None,
                seconds_since_heartbeat=None,
                retry_count=retry_count,
                error_message=None,
            )

        # Check for failed
        if status == "FAILED":
            error_msg = batch_status.get("statusReason") if batch_status else "Unknown error"
            return JobHealthReport(
                service_id=service_id,
                job_id=job_id,
                health=JobHealth.FAILED,
                batch_status=status,
                last_heartbeat=None,
                seconds_since_heartbeat=None,
                retry_count=retry_count,
                error_message=error_msg,
            )

        # For running jobs, check heartbeat
        if status == "RUNNING":
            last_heartbeat = None
            seconds_since = None

            if service_status:
                # Check ended_at (service completed)
                if service_status.get("status") == "COMPLETED":
                    return JobHealthReport(
                        service_id=service_id,
                        job_id=job_id,
                        health=JobHealth.HEALTHY,
                        batch_status=status,
                        last_heartbeat=None,
                        seconds_since_heartbeat=None,
                        retry_count=retry_count,
                        error_message=None,
                    )

                # Check frames_processed as heartbeat proxy
                frames = service_status.get("frames_processed", 0)
                ended_at = service_status.get("ended_at")

                if ended_at:
                    last_heartbeat = datetime.fromisoformat(ended_at.replace("Z", "+00:00")).replace(tzinfo=None)
                    seconds_since = int((now - last_heartbeat).total_seconds())

            # Check job record heartbeat
            if job_record.get("last_heartbeat"):
                hb = datetime.fromisoformat(job_record["last_heartbeat"].replace("Z", "+00:00")).replace(tzinfo=None)
                if last_heartbeat is None or hb > last_heartbeat:
                    last_heartbeat = hb
                    seconds_since = int((now - last_heartbeat).total_seconds())

            # Determine if stuck
            if seconds_since is not None and seconds_since > self.config.heartbeat_timeout_seconds:
                return JobHealthReport(
                    service_id=service_id,
                    job_id=job_id,
                    health=JobHealth.STUCK,
                    batch_status=status,
                    last_heartbeat=last_heartbeat,
                    seconds_since_heartbeat=seconds_since,
                    retry_count=retry_count,
                    error_message=f"No heartbeat for {seconds_since}s",
                )

            # Check timeout
            started_at = batch_status.get("startedAt") if batch_status else None
            if started_at:
                started = datetime.fromtimestamp(started_at / 1000)
                running_minutes = (now - started).total_seconds() / 60
                if running_minutes > self.config.job_timeout_minutes:
                    return JobHealthReport(
                        service_id=service_id,
                        job_id=job_id,
                        health=JobHealth.TIMEOUT,
                        batch_status=status,
                        last_heartbeat=last_heartbeat,
                        seconds_since_heartbeat=seconds_since,
                        retry_count=retry_count,
                        error_message=f"Running for {running_minutes:.0f} minutes",
                    )

            # Running and healthy
            return JobHealthReport(
                service_id=service_id,
                job_id=job_id,
                health=JobHealth.HEALTHY,
                batch_status=status,
                last_heartbeat=last_heartbeat,
                seconds_since_heartbeat=seconds_since,
                retry_count=retry_count,
                error_message=None,
            )

        # Pending/Starting states
        return JobHealthReport(
            service_id=service_id,
            job_id=job_id,
            health=JobHealth.HEALTHY,
            batch_status=status,
            last_heartbeat=None,
            seconds_since_heartbeat=None,
            retry_count=retry_count,
            error_message=None,
        )

    def _restart_job(self, match_id: str, job_record: Dict) -> Optional[str]:
        """
        Restart a failed/stuck job.

        Returns:
            New job ID if restarted, None if max retries exceeded
        """
        service_id = job_record["service_id"]
        retry_count = job_record.get("retry_count", 0) + 1

        if retry_count > self.config.max_retries:
            logger.warning(
                f"Job {service_id} exceeded max retries ({self.config.max_retries}), not restarting"
            )
            return None

        # Determine queue
        service_type = job_record.get("service_type", "")
        cpu_services = {"hls_metadata", "camera_view"}
        queue = self.config.cpu_job_queue if service_type in cpu_services else self.config.gpu_job_queue

        # Get original job to extract parameters
        original_job_id = job_record["job_id"]

        try:
            # Get original job details
            response = self.batch_client.describe_jobs(jobs=[original_job_id])
            if not response.get("jobs"):
                logger.error(f"Original job {original_job_id} not found")
                return None

            original_job = response["jobs"][0]

            # Submit new job with same parameters
            job_name = f"{match_id}_{service_id}_retry{retry_count}"

            submit_args = {
                "jobName": job_name,
                "jobQueue": queue,
                "jobDefinition": original_job["jobDefinition"],
                "parameters": original_job.get("parameters", {}),
            }

            if "containerOverrides" in original_job:
                submit_args["containerOverrides"] = original_job["containerOverrides"]

            new_response = self.batch_client.submit_job(**submit_args)
            new_job_id = new_response["jobId"]

            logger.info(f"Restarted job {service_id}: {original_job_id} → {new_job_id} (retry {retry_count})")

            # Update DynamoDB record
            table = self.dynamodb.Table(self.config.dynamodb_table)
            table.update_item(
                Key={
                    "pk": f"{match_id}#jobs",
                    "sk": service_id
                },
                UpdateExpression="SET job_id = :jid, #status = :status, retry_count = :rc, restarted_at = :ra",
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":jid": new_job_id,
                    ":status": "RESTARTED",
                    ":rc": retry_count,
                    ":ra": datetime.utcnow().isoformat() + "Z",
                }
            )

            return new_job_id

        except Exception as e:
            logger.error(f"Failed to restart job {service_id}: {e}")
            return None

    def _terminate_job(self, job_id: str, reason: str):
        """Terminate a running job"""
        try:
            self.batch_client.terminate_job(
                jobId=job_id,
                reason=reason
            )
            logger.info(f"Terminated job {job_id}: {reason}")
        except Exception as e:
            logger.error(f"Failed to terminate job {job_id}: {e}")

    def _send_alert(self, match_id: str, message: str, severity: str = "WARNING"):
        """Send alert via SNS"""
        if not self.sns_client or not self.config.sns_topic_arn:
            logger.info(f"Alert (no SNS configured): [{severity}] {message}")
            return

        try:
            self.sns_client.publish(
                TopicArn=self.config.sns_topic_arn,
                Subject=f"[{severity}] Inference Job Alert - {match_id}",
                Message=json.dumps({
                    "match_id": match_id,
                    "severity": severity,
                    "message": message,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            )
            logger.info(f"Alert sent: {message}")
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")

    def check_match(self, match_id: str, auto_restart: bool = False) -> MatchHealthReport:
        """
        Check health of all jobs for a match.

        Args:
            match_id: Match identifier
            auto_restart: If True, automatically restart failed/stuck jobs

        Returns:
            MatchHealthReport with status of all jobs
        """
        now = datetime.utcnow()
        actions_taken = []

        # Get job records from DynamoDB
        job_records = self._get_jobs_for_match(match_id)

        if not job_records:
            return MatchHealthReport(
                match_id=match_id,
                overall_health=JobHealth.UNKNOWN,
                job_count=0,
                healthy_count=0,
                failed_count=0,
                stuck_count=0,
                completed_count=0,
                jobs=[],
                actions_taken=[],
            )

        # Get batch status for all jobs
        job_ids = [r["job_id"] for r in job_records]
        batch_statuses = self._get_batch_job_status(job_ids)

        # Check each job
        job_reports = []
        for job_record in job_records:
            service_id = job_record["service_id"]
            job_id = job_record["job_id"]

            # Get batch status
            batch_status = batch_statuses.get(job_id)

            # Get service status (for heartbeat)
            service_status = self._get_service_status(match_id, service_id)

            # Check health
            report = self._check_job_health(job_record, batch_status, service_status, now)

            # Handle unhealthy jobs
            if auto_restart and report.health in (JobHealth.FAILED, JobHealth.STUCK, JobHealth.TIMEOUT):
                # Terminate stuck/timeout jobs first
                if report.health in (JobHealth.STUCK, JobHealth.TIMEOUT):
                    self._terminate_job(job_id, f"Job {report.health.value}")

                # Try to restart
                new_job_id = self._restart_job(match_id, job_record)
                if new_job_id:
                    report.action_taken = f"Restarted as {new_job_id}"
                    actions_taken.append(f"Restarted {service_id}: {job_id} → {new_job_id}")
                else:
                    report.action_taken = "Max retries exceeded - needs manual intervention"
                    actions_taken.append(f"ALERT: {service_id} exceeded max retries")
                    self._send_alert(
                        match_id,
                        f"Job {service_id} failed after {report.retry_count} retries: {report.error_message}",
                        severity="ERROR"
                    )

            job_reports.append(report)

        # Calculate summary
        healthy_count = sum(1 for r in job_reports if r.health == JobHealth.HEALTHY)
        failed_count = sum(1 for r in job_reports if r.health == JobHealth.FAILED)
        stuck_count = sum(1 for r in job_reports if r.health in (JobHealth.STUCK, JobHealth.TIMEOUT))
        completed_count = sum(1 for r in job_reports if r.batch_status == "SUCCEEDED")

        # Determine overall health
        if failed_count > 0 or stuck_count > 0:
            overall = JobHealth.FAILED
        elif healthy_count == len(job_reports):
            overall = JobHealth.HEALTHY
        else:
            overall = JobHealth.UNKNOWN

        return MatchHealthReport(
            match_id=match_id,
            overall_health=overall,
            job_count=len(job_reports),
            healthy_count=healthy_count,
            failed_count=failed_count,
            stuck_count=stuck_count,
            completed_count=completed_count,
            jobs=job_reports,
            actions_taken=actions_taken,
        )

    def check_all_active_matches(self, auto_restart: bool = True) -> List[MatchHealthReport]:
        """
        Check all active matches.

        Returns:
            List of health reports for all active matches
        """
        # Query all active matches
        table = self.dynamodb.Table(self.config.dynamodb_table)

        # Scan for active matches (status != COMPLETED and status != FAILED)
        response = table.scan(
            FilterExpression="contains(pk, :match) AND #status = :running",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":match": "#match",
                ":running": "RUNNING",
            }
        )

        reports = []
        for item in response.get("Items", []):
            match_id = item.get("match_id")
            if match_id:
                report = self.check_match(match_id, auto_restart=auto_restart)
                reports.append(report)

        return reports


# =============================================================================
# LAMBDA HANDLER
# =============================================================================

def monitor_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for monitoring jobs.

    Triggered by EventBridge rule (e.g., every 1 minute).

    Event format:
    {
        "match_id": "match_123"  # Optional - if omitted, checks all active matches
    }
    """
    try:
        monitor = JobMonitor()

        match_id = event.get("match_id")

        if match_id:
            # Check specific match
            report = monitor.check_match(match_id, auto_restart=True)
            reports = [report]
        else:
            # Check all active matches
            reports = monitor.check_all_active_matches(auto_restart=True)

        # Convert to response
        results = []
        for report in reports:
            results.append({
                "match_id": report.match_id,
                "overall_health": report.overall_health.value,
                "job_count": report.job_count,
                "healthy_count": report.healthy_count,
                "failed_count": report.failed_count,
                "stuck_count": report.stuck_count,
                "completed_count": report.completed_count,
                "actions_taken": report.actions_taken,
            })

        return {
            "statusCode": 200,
            "body": json.dumps({
                "checked_at": datetime.utcnow().isoformat() + "Z",
                "matches_checked": len(results),
                "results": results,
            })
        }

    except Exception as e:
        logger.exception(f"Monitor error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point"""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Job Monitor")
    parser.add_argument("--match-id", help="Match to check (omit for all active)")
    parser.add_argument("--auto-restart", action="store_true", help="Automatically restart failed jobs")
    parser.add_argument("--no-restart", action="store_true", help="Don't restart (just report)")

    args = parser.parse_args()

    monitor = JobMonitor()
    auto_restart = args.auto_restart and not args.no_restart

    if args.match_id:
        report = monitor.check_match(args.match_id, auto_restart=auto_restart)
        reports = [report]
    else:
        reports = monitor.check_all_active_matches(auto_restart=auto_restart)

    # Print results
    for report in reports:
        print(f"\n{'='*60}")
        print(f"Match: {report.match_id}")
        print(f"Overall Health: {report.overall_health.value}")
        print(f"Jobs: {report.job_count} total, {report.healthy_count} healthy, "
              f"{report.failed_count} failed, {report.stuck_count} stuck, "
              f"{report.completed_count} completed")

        if report.actions_taken:
            print(f"\nActions taken:")
            for action in report.actions_taken:
                print(f"  - {action}")

        print(f"\nJob Details:")
        for job in report.jobs:
            status_icon = "✓" if job.health == JobHealth.HEALTHY else "✗"
            print(f"  {status_icon} {job.service_id}: {job.batch_status} ({job.health.value})")
            if job.error_message:
                print(f"      Error: {job.error_message}")
            if job.action_taken:
                print(f"      Action: {job.action_taken}")


if __name__ == "__main__":
    main()
