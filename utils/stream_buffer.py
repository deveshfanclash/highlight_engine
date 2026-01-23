"""
Stream Buffer

Thread-safe frame buffer with backpressure control for decoupling
frame production (FFmpeg/OpenCV) from frame consumption (inference).

Usage:
    buffer = StreamBuffer(max_size=30, mode=BufferMode.FIFO)

    # Producer thread
    for frame in frame_source:
        if not buffer.put(frame):
            logger.warning("Buffer full, backpressure applied")

    # Consumer thread
    while True:
        frame = buffer.get(timeout=1.0)
        if frame is None:
            break
        process(frame)
"""

import logging
from queue import Queue, Full, Empty
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Generic, TypeVar, Iterator
from threading import Event

logger = logging.getLogger(__name__)

T = TypeVar('T')


class BufferMode(Enum):
    """Buffer behavior when full."""
    FIFO = "fifo"          # Block/reject new frames when full (preserve all)
    DROP_OLD = "drop_old"  # Drop oldest frame to make room (real-time)


@dataclass
class StreamBuffer(Generic[T]):
    """
    Thread-safe frame buffer with configurable backpressure behavior.

    Attributes:
        max_size: Maximum frames to buffer (default: 30)
        mode: FIFO (block when full) or DROP_OLD (drop oldest for real-time)

    Example:
        # For VOD processing (preserve all frames)
        buffer = StreamBuffer(max_size=30, mode=BufferMode.FIFO)

        # For real-time streaming (always use latest)
        buffer = StreamBuffer(max_size=10, mode=BufferMode.DROP_OLD)
    """
    max_size: int = 30
    mode: BufferMode = BufferMode.FIFO

    _queue: Queue = field(init=False, repr=False)
    _stopped: Event = field(init=False, repr=False)
    _items_put: int = field(init=False, default=0)
    _items_dropped: int = field(init=False, default=0)

    def __post_init__(self):
        self._queue = Queue(maxsize=self.max_size)
        self._stopped = Event()
        self._items_put = 0
        self._items_dropped = 0

    def put(self, item: T, timeout: float = 1.0) -> bool:
        """
        Put item in buffer.

        Args:
            item: Frame or data to buffer
            timeout: Seconds to wait if buffer full (FIFO mode)

        Returns:
            True if item was added, False if buffer full/stopped
        """
        if self._stopped.is_set():
            return False

        try:
            if self.mode == BufferMode.DROP_OLD and self._queue.full():
                # Drop oldest to make room
                try:
                    self._queue.get_nowait()
                    self._items_dropped += 1
                except Empty:
                    pass

            self._queue.put(item, timeout=timeout)
            self._items_put += 1
            return True

        except Full:
            return False

    def get(self, timeout: float = 1.0) -> Optional[T]:
        """
        Get item from buffer.

        Args:
            timeout: Seconds to wait for item

        Returns:
            Item or None if timeout/stopped
        """
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def get_nowait(self) -> Optional[T]:
        """Get item without blocking."""
        try:
            return self._queue.get_nowait()
        except Empty:
            return None

    def peek(self) -> Optional[T]:
        """Look at next item without removing (not thread-safe for value)."""
        if self._queue.empty():
            return None
        # Note: This is a simplification - true peek would need mutex
        return None  # Queue doesn't support peek, return None

    def clear(self):
        """Clear all items from buffer."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def stop(self):
        """Signal buffer to stop (get() will return None)."""
        self._stopped.set()

    def iterate(self, timeout: float = 1.0) -> Iterator[T]:
        """
        Iterate over buffer items until stopped.

        Usage:
            for frame in buffer.iterate():
                process(frame)
        """
        while not self._stopped.is_set():
            item = self.get(timeout=timeout)
            if item is not None:
                yield item
            elif self._stopped.is_set():
                break

    @property
    def is_full(self) -> bool:
        return self._queue.full()

    @property
    def is_empty(self) -> bool:
        return self._queue.empty()

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def items_put(self) -> int:
        """Total items added to buffer."""
        return self._items_put

    @property
    def items_dropped(self) -> int:
        """Items dropped due to DROP_OLD mode."""
        return self._items_dropped

    @property
    def is_stopped(self) -> bool:
        return self._stopped.is_set()


class BufferedFrameProvider:
    """
    Wraps a frame source with a StreamBuffer for async processing.

    Runs frame extraction in a background thread, buffers frames,
    and provides iteration for the consumer.

    Usage:
        provider = BufferedFrameProvider(
            frame_source=frame_provider.frames(),
            buffer_size=30,
            mode=BufferMode.FIFO
        )
        provider.start()

        for frame_packet in provider.iterate():
            process(frame_packet)

        provider.stop()
    """

    def __init__(
        self,
        frame_source: Iterator[T],
        buffer_size: int = 30,
        mode: BufferMode = BufferMode.FIFO,
    ):
        self.frame_source = frame_source
        self.buffer = StreamBuffer(max_size=buffer_size, mode=mode)
        self._producer_thread = None
        self._running = False

    def _producer_loop(self):
        """Background thread that reads frames into buffer."""
        try:
            for frame in self.frame_source:
                if not self._running:
                    break

                success = self.buffer.put(frame, timeout=1.0)
                if not success and self._running:
                    logger.warning("Buffer full, frame dropped")

        except Exception as e:
            logger.error(f"Error in frame producer: {e}")
        finally:
            self.buffer.stop()
            logger.debug("Frame producer stopped")

    def start(self):
        """Start background frame extraction."""
        from threading import Thread

        self._running = True
        self._producer_thread = Thread(target=self._producer_loop, daemon=True)
        self._producer_thread.start()
        logger.info(f"BufferedFrameProvider started (buffer_size={self.buffer.max_size})")

    def stop(self):
        """Stop frame extraction."""
        self._running = False
        self.buffer.stop()

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5.0)

        logger.info(
            f"BufferedFrameProvider stopped: "
            f"{self.buffer.items_put} frames buffered, "
            f"{self.buffer.items_dropped} dropped"
        )

    def iterate(self, timeout: float = 1.0) -> Iterator[T]:
        """Iterate over buffered frames."""
        return self.buffer.iterate(timeout=timeout)

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "current_size": self.buffer.size,
            "max_size": self.buffer.max_size,
            "items_put": self.buffer.items_put,
            "items_dropped": self.buffer.items_dropped,
            "is_full": self.buffer.is_full,
        }


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import time
    from threading import Thread

    print("Testing StreamBuffer...")

    # Test basic operations
    buffer = StreamBuffer(max_size=5, mode=BufferMode.FIFO)

    # Put items
    for i in range(5):
        assert buffer.put(i), f"Failed to put item {i}"
    assert buffer.is_full, "Buffer should be full"
    assert buffer.size == 5, f"Size should be 5, got {buffer.size}"

    # Get items
    for i in range(5):
        item = buffer.get(timeout=0.1)
        assert item == i, f"Expected {i}, got {item}"
    assert buffer.is_empty, "Buffer should be empty"
    print("  FIFO mode works")

    # Test DROP_OLD mode
    buffer = StreamBuffer(max_size=3, mode=BufferMode.DROP_OLD)
    for i in range(10):
        buffer.put(i)

    # Should have last 3 items (7, 8, 9)
    items = []
    while not buffer.is_empty:
        items.append(buffer.get_nowait())
    assert items == [7, 8, 9], f"Expected [7, 8, 9], got {items}"
    assert buffer.items_dropped == 7, f"Expected 7 dropped, got {buffer.items_dropped}"
    print("  DROP_OLD mode works")

    # Test producer/consumer with BufferedFrameProvider
    print("\nTesting BufferedFrameProvider...")

    def slow_producer():
        for i in range(20):
            time.sleep(0.01)
            yield i

    provider = BufferedFrameProvider(
        frame_source=slow_producer(),
        buffer_size=5,
        mode=BufferMode.FIFO
    )
    provider.start()

    consumed = []
    for item in provider.iterate(timeout=0.5):
        consumed.append(item)
        if len(consumed) >= 20:
            break

    provider.stop()
    assert consumed == list(range(20)), f"Expected 0-19, got {consumed}"
    print(f"  BufferedFrameProvider works: {provider.stats}")

    print("\nAll tests passed!")
