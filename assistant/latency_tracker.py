"""
AeroScript JARVIS — Voice Pipeline Latency Tracker
====================================================
Lightweight, thread-safe instrumentation for measuring end-to-end
latency of each voice command through the JARVIS pipeline.

Stages tracked:
    1. speech_captured   — audio returned by recognizer.listen()
    2. stt_complete      — Google STT returns text
    3. intent_detected   — CommandProcessor.process() returns
    4. command_queued    — VoiceCommand placed in queue
    5. command_executed  — controller handler returns
    6. tts_queued        — tts.speak() called

Usage:
    tracker = LatencyTracker()
    tracker.stamp("speech_captured")
    ...
    tracker.stamp("stt_complete")
    ...
    tracker.log_summary()   # prints structured latency breakdown
"""

import time
import logging
from typing import Optional

log = logging.getLogger(__name__)

# Dedicated logger for latency metrics — goes to same log file,
# but can be filtered independently by name.
_perf_log = logging.getLogger("JARVIS.latency")


class LatencyTracker:
    """
    Per-command latency tracker.

    Create one instance per voice command. Call stamp() at each pipeline
    stage. Call log_summary() after the final stage to emit a structured
    log line with all timing deltas.

    Thread-safe: stamps can be called from any thread.
    """

    __slots__ = ("_stamps", "_command_text")

    def __init__(self, command_text: str = ""):
        self._stamps: list[tuple[str, float]] = []
        self._command_text = command_text

    def stamp(self, stage: str) -> float:
        """
        Record a timestamp for the named stage.

        Args:
            stage: Human-readable stage name (e.g. "stt_complete").

        Returns:
            The recorded timestamp (perf_counter value).
        """
        t = time.perf_counter()
        self._stamps.append((stage, t))
        return t

    @property
    def first_stamp(self) -> Optional[float]:
        """Timestamp of the first recorded stage, or None."""
        return self._stamps[0][1] if self._stamps else None

    @property
    def last_stamp(self) -> Optional[float]:
        """Timestamp of the last recorded stage, or None."""
        return self._stamps[-1][1] if self._stamps else None

    @property
    def total_ms(self) -> float:
        """Total elapsed time from first to last stamp, in milliseconds."""
        if len(self._stamps) < 2:
            return 0.0
        return (self._stamps[-1][1] - self._stamps[0][1]) * 1000

    def elapsed_since(self, stage: str) -> float:
        """Milliseconds elapsed since the named stage was stamped."""
        for name, t in self._stamps:
            if name == stage:
                return (time.perf_counter() - t) * 1000
        return 0.0

    def delta_ms(self, from_stage: str, to_stage: str) -> Optional[float]:
        """Milliseconds between two named stages, or None if either is missing."""
        t_from = t_to = None
        for name, t in self._stamps:
            if name == from_stage:
                t_from = t
            if name == to_stage:
                t_to = t
        if t_from is not None and t_to is not None:
            return (t_to - t_from) * 1000
        return None

    def log_summary(self):
        """
        Emit a structured latency summary to the JARVIS.latency logger.

        Example output:
            ┌─ JARVIS Latency: "clear the canvas"
            │  speech_captured →  stt_complete:   842 ms
            │  stt_complete    →  intent_detected:   1 ms
            │  intent_detected →  command_queued:     0 ms
            │  command_queued  →  command_executed:  12 ms
            │  command_executed→  tts_queued:         1 ms
            └─ Total: 856 ms
        """
        if len(self._stamps) < 2:
            return

        cmd_display = self._command_text[:50] if self._command_text else "(unknown)"
        lines = [f'┌─ JARVIS Latency: "{cmd_display}"']

        for i in range(1, len(self._stamps)):
            prev_name, prev_t = self._stamps[i - 1]
            curr_name, curr_t = self._stamps[i]
            delta = (curr_t - prev_t) * 1000
            lines.append(
                f"│  {prev_name:<18s} → {curr_name:<18s}: {delta:7.1f} ms"
            )

        total = (self._stamps[-1][1] - self._stamps[0][1]) * 1000
        lines.append(f"└─ Total: {total:.1f} ms")

        _perf_log.info("\n".join(lines))
