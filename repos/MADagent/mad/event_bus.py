"""Event bus and scheduler.

Routes typed events to subsystems, enforces pacing, provides
backpressure and priority ordering.
"""

from __future__ import annotations

import heapq
import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from mad.models import Event, EventType

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedEvent:
    priority: float
    sequence: int = field(compare=True)
    event: Event = field(compare=False)


EventHandler = Callable[[Event], None]


class EventBus:
    """Priority-based event bus with subscriber routing.

    Events are dispatched in priority order (lower = higher priority).
    Subscribers register for specific event types.
    """

    def __init__(self, max_queue_size: int = 1000) -> None:
        self._queue: list[PrioritizedEvent] = []
        self._sequence = 0
        self._subscribers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._global_subscribers: list[EventHandler] = []
        self._lock = threading.Lock()
        self._max_queue_size = max_queue_size
        self._processed_count = 0

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        self._subscribers[event_type].append(handler)

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler for all event types."""
        self._global_subscribers.append(handler)

    def publish(self, event: Event, priority: float = 0.5) -> bool:
        """Publish an event to the bus. Returns False if backpressure is applied."""
        assert isinstance(event, Event), f"event must be an Event, got {type(event).__name__}"
        with self._lock:
            if len(self._queue) >= self._max_queue_size:
                logger.warning("Event bus backpressure: queue full")
                return False
            pe = PrioritizedEvent(
                priority=priority,
                sequence=self._sequence,
                event=event,
            )
            heapq.heappush(self._queue, pe)
            self._sequence += 1
            return True

    def process_one(self) -> Event | None:
        """Process the highest-priority event. Returns the event or None."""
        with self._lock:
            if not self._queue:
                return None
            pe = heapq.heappop(self._queue)

        event = pe.event
        self._dispatch(event)
        self._processed_count += 1
        return event

    def process_all(self) -> int:
        """Process all queued events. Returns count processed.

        Bounded to max_queue_size * 2 iterations to prevent infinite loops
        if handlers publish new events during dispatch.
        """
        max_iterations = self._max_queue_size * 2
        count = 0
        for _ in range(max_iterations):
            event = self.process_one()
            if event is None:
                break
            count += 1
        else:
            logger.warning(
                "EventBus.process_all hit iteration limit (%d); "
                "%d events may remain queued",
                max_iterations,
                self.pending_count,
            )
        return count

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to registered handlers."""
        for handler in self._global_subscribers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Global handler error: {e}")

        for handler in self._subscribers.get(event.type, []):
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.type}: {e}")

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def processed_count(self) -> int:
        return self._processed_count

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
