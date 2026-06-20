"""
Simulation Clock – Manages simulated time progression.
Supports pause, resume, speed control, and wall-time tracking.
"""

import time


class SimulationClock:
    """Event-driven simulation clock with pause/resume and speed control."""

    def __init__(self, dt: float = 0.01, speed: float = 1.0):
        self.dt = dt
        self.speed = speed
        self.current_time = 0.0
        self.start_wall_time = 0.0
        self._paused = False
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────
    def start(self):
        """Start or resume the clock."""
        self._running = True
        self._paused = False
        self.start_wall_time = time.perf_counter()

    def pause(self):
        """Pause time progression."""
        self._paused = True

    def resume(self):
        """Resume from pause."""
        self._paused = False

    def reset(self):
        """Reset clock to t = 0."""
        self.current_time = 0.0
        self._paused = False
        self._running = False
        self.start_wall_time = 0.0

    def stop(self):
        """Stop the clock entirely."""
        self._running = False

    # ── Time Advancement ───────────────────────────────────────────────
    def tick(self) -> float:
        """
        Advance simulation time by one step (dt * speed).
        Returns the new current time.
        """
        if self._paused or not self._running:
            return self.current_time
        self.current_time += self.dt * self.speed
        return self.current_time

    # ── Properties ─────────────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self._running and not self._paused

    @property
    def elapsed_wall_time(self) -> float:
        """Wall-clock time since start (seconds)."""
        if self.start_wall_time == 0.0:
            return 0.0
        return time.perf_counter() - self.start_wall_time

    def set_speed(self, speed: float):
        """Set time acceleration factor (1.0 = real-time)."""
        self.speed = max(0.01, speed)

    def __repr__(self):
        state = "running" if self.is_running else ("paused" if self._paused else "stopped")
        return f"SimulationClock(t={self.current_time:.4f}s, dt={self.dt}, speed={self.speed}x, {state})"
