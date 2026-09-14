"""Tests for event bus and scheduling."""

import pytest

from mad.event_bus import EventBus
from mad.models import Event, EventType


class TestEventBus:
    def test_publish_and_process(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.USER_INPUT, lambda e: received.append(e))
        event = Event(type=EventType.USER_INPUT, payload={"text": "hello"})
        bus.publish(event)
        bus.process_one()
        assert len(received) == 1
        assert received[0].payload["text"] == "hello"

    def test_priority_ordering(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e.payload["n"]))

        # Publish in reverse priority (lower number = higher priority)
        bus.publish(Event(type=EventType.SYSTEM, payload={"n": 3}), priority=0.9)
        bus.publish(Event(type=EventType.SYSTEM, payload={"n": 1}), priority=0.1)
        bus.publish(Event(type=EventType.SYSTEM, payload={"n": 2}), priority=0.5)

        bus.process_all()
        assert received == [1, 2, 3]

    def test_backpressure(self):
        bus = EventBus(max_queue_size=2)
        bus.publish(Event(type=EventType.SYSTEM, payload={}))
        bus.publish(Event(type=EventType.SYSTEM, payload={}))
        result = bus.publish(Event(type=EventType.SYSTEM, payload={}))
        assert result is False

    def test_process_all(self):
        bus = EventBus()
        count = 0

        def inc(e):
            nonlocal count
            count += 1

        bus.subscribe_all(inc)
        for i in range(5):
            bus.publish(Event(type=EventType.SYSTEM, payload={"n": i}))
        processed = bus.process_all()
        assert processed == 5
        assert count == 5

    def test_global_subscriber(self):
        bus = EventBus()
        received = []
        bus.subscribe_all(lambda e: received.append(e.type))
        bus.publish(Event(type=EventType.USER_INPUT, payload={}))
        bus.publish(Event(type=EventType.SYSTEM, payload={}))
        bus.process_all()
        assert EventType.USER_INPUT in received
        assert EventType.SYSTEM in received

    def test_pending_count(self):
        bus = EventBus()
        assert bus.pending_count == 0
        bus.publish(Event(type=EventType.SYSTEM, payload={}))
        assert bus.pending_count == 1
        bus.process_one()
        assert bus.pending_count == 0
