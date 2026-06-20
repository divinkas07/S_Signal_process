"""
Event Bus – Publish/Subscribe communication between decoupled modules.
Allows any module to broadcast events without knowing the consumers.
"""

from collections import defaultdict
from typing import Callable, Any


class EventBus:
    """Lightweight pub/sub event system for inter-module communication."""

    _instance = None  # Singleton for global access

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._event_log: list[tuple[str, Any]] = []
        self._logging_enabled = False

    # ── Singleton Access ───────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> "EventBus":
        """Get or create the global EventBus instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton (for testing)."""
        cls._instance = None

    # ── Subscribe / Unsubscribe ────────────────────────────────────────
    def subscribe(self, event_name: str, callback: Callable):
        """Register a callback for an event type."""
        if callback not in self._subscribers[event_name]:
            self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback: Callable):
        """Remove a callback from an event type."""
        if event_name in self._subscribers:
            self._subscribers[event_name] = [
                cb for cb in self._subscribers[event_name] if cb != callback
            ]

    # ── Publish ────────────────────────────────────────────────────────
    def publish(self, event_name: str, data: Any = None):
        """
        Broadcast an event to all subscribers.
        Calls each subscriber synchronously in registration order.
        """
        if self._logging_enabled:
            self._event_log.append((event_name, data))

        for callback in self._subscribers.get(event_name, []):
            try:
                callback(data)
            except Exception as e:
                print(f"[EventBus] Error in handler for '{event_name}': {e}")

    # ── Management ─────────────────────────────────────────────────────
    def clear(self):
        """Remove all subscriptions."""
        self._subscribers.clear()
        self._event_log.clear()

    def enable_logging(self, enabled: bool = True):
        """Enable/disable event logging for debugging."""
        self._logging_enabled = enabled
        if not enabled:
            self._event_log.clear()

    def get_log(self) -> list[tuple[str, Any]]:
        """Return the event log (only if logging is enabled)."""
        return list(self._event_log)

    @property
    def subscriber_count(self) -> int:
        """Total number of active subscriptions."""
        return sum(len(cbs) for cbs in self._subscribers.values())

    def __repr__(self):
        events = list(self._subscribers.keys())
        return f"EventBus(events={events}, total_subs={self.subscriber_count})"


# ─── Standard Event Names ────────────────────────────────────────────────
class Events:
    """Standard event names used across the simulator."""
    TICK = "sim.tick"
    SIM_START = "sim.start"
    SIM_STOP = "sim.stop"
    SIM_PAUSE = "sim.pause"
    SIM_RESET = "sim.reset"
    SATELLITE_UPDATED = "satellite.updated"
    STATION_UPDATED = "station.updated"
    SIGNAL_GENERATED = "signal.generated"
    CHANNEL_APPLIED = "channel.applied"
    AOA_ESTIMATED = "aoa.estimated"
    METRICS_UPDATED = "metrics.updated"
    LINK_ESTABLISHED = "link.established"
    LINK_LOST = "link.lost"
    DEVICE_ADDED = "device.added"
    DEVICE_REMOVED = "device.removed"
