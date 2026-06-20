"""
Event-Driven Scheduler – Priority queue for timed simulation events.
Allows scheduling of callbacks at specific simulation times.
"""

import heapq
from typing import Callable, Any


class ScheduledEvent:
    """A callback scheduled to execute at a specific simulation time."""
    __slots__ = ("time", "priority", "callback", "args", "_counter")
    _global_counter = 0

    def __init__(self, time: float, callback: Callable, args: tuple = (), priority: int = 0):
        self.time = time
        self.priority = priority
        self.callback = callback
        self.args = args
        ScheduledEvent._global_counter += 1
        self._counter = ScheduledEvent._global_counter

    def __lt__(self, other):
        if self.time != other.time:
            return self.time < other.time
        if self.priority != other.priority:
            return self.priority < other.priority
        return self._counter < other._counter


class Scheduler:
    """
    Event-driven simulation scheduler using a priority queue.
    Events are executed in chronological order.
    """

    def __init__(self):
        self._queue: list[ScheduledEvent] = []
        self._current_time: float = 0.0

    def schedule(self, t: float, callback: Callable, args: tuple = (), priority: int = 0):
        """
        Schedule a callback to fire at simulation time t.
        Lower priority values execute first for same-time events.
        """
        event = ScheduledEvent(t, callback, args, priority)
        heapq.heappush(self._queue, event)

    def schedule_recurring(self, start: float, interval: float, callback: Callable,
                          end: float = float("inf"), args: tuple = ()):
        """Schedule a callback to fire repeatedly at fixed intervals."""
        t = start
        while t <= end and t < 1e9:  # Safety limit
            self.schedule(t, callback, args)
            t += interval

    def step(self) -> bool:
        """
        Execute the next event in the queue.
        Returns True if an event was processed, False if queue is empty.
        """
        if not self._queue:
            return False
        event = heapq.heappop(self._queue)
        self._current_time = event.time
        event.callback(*event.args)
        return True

    def run_until(self, t_end: float):
        """Execute all events up to time t_end."""
        while self._queue and self._queue[0].time <= t_end:
            self.step()
        self._current_time = t_end

    @property
    def current_time(self) -> float:
        return self._current_time

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def next_event_time(self) -> float:
        if self._queue:
            return self._queue[0].time
        return float("inf")

    def clear(self):
        """Remove all scheduled events."""
        self._queue.clear()
        self._current_time = 0.0

    def __repr__(self):
        return f"Scheduler(pending={self.pending_count}, t={self._current_time:.4f}s)"
