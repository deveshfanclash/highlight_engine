#!/usr/bin/env python3
"""
Setup Config Script
Reads GAME_CONFIG from .env file, converts JSON to YAML, and saves to config folder.

Usage:
    python setup_config.py
    python setup_config.py --env-file /path/to/.env
    python setup_config.py --output-dir /path/to/config
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_env_file(env_path: str) -> dict:
    """Load environment variables from .env file."""
    env_vars = {}
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            # Parse key=value
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    return env_vars


def json_to_yaml(data: dict, indent: int = 0) -> str:
    """Convert a dictionary to YAML string manually (no external deps)."""
    lines = []
    prefix = '  ' * indent

    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(json_to_yaml(value, indent + 1))
        elif isinstance(value, list):
            lines.append(f"{prefix}{key}:")
            for item in value:
                if isinstance(item, dict):
                    # First key on same line as dash
                    first_key = True
                    for k, v in item.items():
                        if first_key:
                            if isinstance(v, dict):
                                lines.append(f"{prefix}  - {k}:")
                                lines.append(json_to_yaml(v, indent + 3))
                            elif isinstance(v, list):
                                lines.append(f"{prefix}  - {k}:")
                                for sub_item in v:
                                    lines.append(f"{prefix}      - {format_value(sub_item)}")
                            else:
                                lines.append(f"{prefix}  - {k}: {format_value(v)}")
                            first_key = False
                        else:
                            if isinstance(v, dict):
                                lines.append(f"{prefix}    {k}:")
                                lines.append(json_to_yaml(v, indent + 3))
                            elif isinstance(v, list):
                                lines.append(f"{prefix}    {k}:")
                                for sub_item in v:
                                    lines.append(f"{prefix}      - {format_value(sub_item)}")
                            else:
                                lines.append(f"{prefix}    {k}: {format_value(v)}")
                else:
                    lines.append(f"{prefix}  - {format_value(item)}")
        else:
            lines.append(f"{prefix}{key}: {format_value(value)}")

    return '\n'.join(lines)


def format_value(value) -> str:
    """Format a value for YAML output."""
    if value is None:
        return 'null'
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, str):
        # Quote strings that might be ambiguous
        if value in ('true', 'false', 'null', 'yes', 'no') or value.isdigit():
            return f'"{value}"'
        # Quote strings with special characters
        if any(c in value for c in ':{}[]&*#?|-<>=!%@\\'):
            return f'"{value}"'
        return value
    else:
        return str(value)


def main():
    parser = argparse.ArgumentParser(description='Convert GAME_CONFIG from .env to YAML')
    parser.add_argument('--env-file', default='.env', help='Path to .env file')
    parser.add_argument('--output-dir', default='config/games', help='Output directory for YAML config')
    parser.add_argument('--output-name', default=None, help='Output filename (default: uses game_id from config)')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent.resolve()
    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = script_dir / env_path

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir

    # Check .env exists
    if not env_path.exists():
        print(f"Error: .env file not found at {env_path}")
        sys.exit(1)

    # Load .env
    print(f"Loading environment from: {env_path}")
    env_vars = load_env_file(str(env_path))

    # Get GAME_CONFIG
    game_config_str = env_vars.get('GAME_CONFIG')
    if not game_config_str:
        print("Error: GAME_CONFIG not found in .env file")
        sys.exit(1)

    # Parse JSON
    try:
        config = json.loads(game_config_str)
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse GAME_CONFIG as JSON: {e}")
        sys.exit(1)

    # Determine output filename
    game_id = config.get('game_id', 'game_config')
    output_name = args.output_name or f"{game_id}.yaml"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    # Generate YAML header
    yaml_content = f"""# =============================================================================
# {config.get('game_name', 'Game')} Configuration
# =============================================================================
# Auto-generated from .env GAME_CONFIG
# =============================================================================

"""

    # Convert to YAML
    yaml_content += json_to_yaml(config)

    # Write YAML file
    with open(output_path, 'w') as f:
        f.write(yaml_content)

    print(f"Config saved to: {output_path}")
    print(f"Game ID: {game_id}")
    print(f"Game Name: {config.get('game_name', 'Unknown')}")
    print(f"Models: {len(config.get('models', []))}")
    print(f"Services: {len(config.get('services', []))}")


if __name__ == '__main__':
    main()
