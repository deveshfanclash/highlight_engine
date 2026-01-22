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
from dataclasses import dataclass
from typing import TypeVar, Generic, Iterator, List

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
    print("  All indices accounted for")

    # Test drop_last
    accumulator_drop = BatchAccumulator(batch_size=8, drop_last=True)
    batches_drop = list(accumulator_drop.batches(iter(items)))
    print(f"\n  With drop_last=True: {len(batches_drop)} batches")
    assert len(batches_drop) == 3, "Should have 3 complete batches"
    print("  Incomplete batch dropped")

    print("\nAll tests passed!")
