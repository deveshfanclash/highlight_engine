"""
Batch Accumulator

Accumulates items into batches for efficient GPU processing.
Processing N images in a batch takes nearly the same time as processing 1 image,
making batching a key optimization for throughput.

This follows the BatchAccumulator pattern from Ultralytics architecture:
- Collects items from an iterator into batches
- Preserves original ordering for result mapping
- Configurable batch size and behavior
"""

import logging
from dataclasses import dataclass, field
from typing import TypeVar, Generic, Iterator, List, Callable, Any, Optional

logger = logging.getLogger(__name__)

T = TypeVar('T')  # Type of items being batched


@dataclass
class Batch(Generic[T]):
    """
    A batch of items with metadata for result mapping.

    Attributes:
        items: List of items in this batch
        indices: Original positions of items (for mapping results back)
        size: Number of items in batch
    """
    items: List[T]
    indices: List[int]

    @property
    def size(self) -> int:
        return len(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class BatchAccumulator(Generic[T]):
    """
    Accumulates items from an iterator into batches for efficient processing.

    Why this exists:
        - GPU processing of 1 image vs 8 images takes nearly the same time
        - Batching can provide 3-8x throughput improvement
        - This class abstracts batch management from service logic

    Usage:
        # Basic usage
        accumulator = BatchAccumulator(batch_size=8)

        for batch in accumulator.batches(frame_iterator):
            # Process batch of 8 frames at once
            results = model.predict([f.frame for f in batch.items])

            # Map results back to original frames
            for packet, result in zip(batch.items, results):
                process_result(packet, result)

        # With frame extraction
        for batch in accumulator.batches(input_handler.iterate()):
            frames = [pkt.frame for pkt in batch.items]
            outputs = model.predict(frames)

            for pkt, output in zip(batch.items, outputs):
                yield pkt, output

    Configuration:
        batch_size: Number of items per batch (default: 8)
        drop_last: If True, drop incomplete final batch (default: False)
        timeout_items: Max items to buffer before forcing yield (for latency control)
    """

    def __init__(
        self,
        batch_size: int = 8,
        drop_last: bool = False,
    ):
        """
        Initialize batch accumulator.

        Args:
            batch_size: Number of items per batch
            drop_last: If True, drop the final batch if it's incomplete
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.batch_size = batch_size
        self.drop_last = drop_last

    def batches(self, items: Iterator[T]) -> Iterator[Batch[T]]:
        """
        Yield batches of items from an iterator.

        Args:
            items: Iterator of items to batch

        Yields:
            Batch objects containing items and their original indices
        """
        batch_items: List[T] = []
        batch_indices: List[int] = []
        idx = 0

        for item in items:
            batch_items.append(item)
            batch_indices.append(idx)
            idx += 1

            # Yield when batch is full
            if len(batch_items) >= self.batch_size:
                yield Batch(items=batch_items, indices=batch_indices)
                batch_items = []
                batch_indices = []

        # Yield remaining items (unless drop_last is True)
        if batch_items and not self.drop_last:
            yield Batch(items=batch_items, indices=batch_indices)

    def process_batched(
        self,
        items: Iterator[T],
        process_fn: Callable[[List[T]], List[Any]],
    ) -> Iterator[tuple]:
        """
        Process items in batches and yield (item, result) pairs.

        This is a convenience method that handles the common pattern of:
        1. Accumulating items into batches
        2. Processing each batch
        3. Yielding (item, result) pairs in order

        Args:
            items: Iterator of items to process
            process_fn: Function that takes a list of items and returns results

        Yields:
            Tuples of (original_item, result) for each item
        """
        for batch in self.batches(items):
            results = process_fn(batch.items)

            # Ensure results match batch size
            if len(results) != batch.size:
                logger.warning(
                    f"Batch size mismatch: expected {batch.size} results, "
                    f"got {len(results)}"
                )
                # Pad or truncate results as needed
                if len(results) < batch.size:
                    results = list(results) + [None] * (batch.size - len(results))
                else:
                    results = results[:batch.size]

            for item, result in zip(batch.items, results):
                yield item, result


# =============================================================================
# SPECIALIZED BATCH ACCUMULATORS
# =============================================================================

class FrameBatchAccumulator(BatchAccumulator):
    """
    Batch accumulator specialized for frame processing.

    Adds frame-specific utilities:
    - Extract frames from packets
    - Track frame numbers for logging
    - Support for frame skipping within batches
    """

    def __init__(
        self,
        batch_size: int = 8,
        drop_last: bool = False,
        frame_extractor: Optional[Callable[[Any], Any]] = None,
    ):
        """
        Initialize frame batch accumulator.

        Args:
            batch_size: Number of frames per batch
            drop_last: If True, drop incomplete final batch
            frame_extractor: Function to extract frame from packet (default: packet.frame)
        """
        super().__init__(batch_size=batch_size, drop_last=drop_last)
        self.frame_extractor = frame_extractor or (lambda pkt: pkt.frame)

    def extract_frames(self, batch: Batch) -> List[Any]:
        """Extract frames from a batch of packets."""
        return [self.frame_extractor(item) for item in batch.items]

    def batches_with_frames(self, items: Iterator[T]) -> Iterator[tuple]:
        """
        Yield (batch, frames) tuples for convenience.

        Args:
            items: Iterator of frame packets

        Yields:
            Tuples of (Batch, List[frames])
        """
        for batch in self.batches(items):
            frames = self.extract_frames(batch)
            yield batch, frames


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def batch_iterate(
    items: Iterator[T],
    batch_size: int = 8,
    drop_last: bool = False,
) -> Iterator[Batch[T]]:
    """
    Convenience function for simple batching.

    Args:
        items: Iterator of items
        batch_size: Items per batch
        drop_last: Drop incomplete final batch

    Yields:
        Batch objects
    """
    accumulator = BatchAccumulator(batch_size=batch_size, drop_last=drop_last)
    yield from accumulator.batches(items)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test basic batching
    print("Testing BatchAccumulator...")

    # Test with list
    items = list(range(25))
    accumulator = BatchAccumulator(batch_size=8)

    batches = list(accumulator.batches(iter(items)))
    print(f"  Input: {len(items)} items, batch_size=8")
    print(f"  Output: {len(batches)} batches")
    for i, batch in enumerate(batches):
        print(f"    Batch {i}: {batch.size} items, indices {batch.indices[0]}-{batch.indices[-1]}")

    # Verify all items processed
    all_indices = []
    for batch in batches:
        all_indices.extend(batch.indices)
    assert all_indices == list(range(25)), "Index mismatch!"
    print("  ✓ All indices accounted for")

    # Test drop_last
    accumulator_drop = BatchAccumulator(batch_size=8, drop_last=True)
    batches_drop = list(accumulator_drop.batches(iter(items)))
    print(f"\n  With drop_last=True: {len(batches_drop)} batches")
    assert len(batches_drop) == 3, "Should have 3 complete batches"
    print("  ✓ Incomplete batch dropped")

    # Test process_batched
    print("\nTesting process_batched...")

    def double_batch(items):
        return [x * 2 for x in items]

    results = list(accumulator.process_batched(iter(range(10)), double_batch))
    expected = [(i, i * 2) for i in range(10)]
    assert results == expected, f"Expected {expected}, got {results}"
    print("  ✓ process_batched works correctly")

    print("\nAll tests passed!")
