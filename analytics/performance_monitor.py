"""
AeroScript 2.0 — Performance Monitor
Real-time FPS, latency, CPU and memory tracking.
"""

import time
import collections
import logging

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil not available — CPU/memory monitoring disabled.")


class PerformanceMonitor:
    """
    Tracks real-time performance metrics.

    Metrics:
        fps         : rolling average over last N frames
        latency_ms  : time from frame read to display (set manually each loop)
        cpu_pct     : system CPU usage %
        mem_mb      : current process memory in MB
    """

    def __init__(self, buffer_size: int = 30):
        self._frame_times: collections.deque = collections.deque(maxlen=buffer_size)
        self._last_tick   = time.perf_counter()
        self._latency_ms  = 0.0

        if _PSUTIL_AVAILABLE:
            self._process = psutil.Process()
        else:
            self._process = None

        # Rate-limit expensive psutil calls
        self._last_system_check = 0.0
        self._cached_cpu  = 0.0
        self._cached_mem  = 0.0

    def tick(self):
        """
        Call once per frame at the start of the loop iteration.
        Records the inter-frame time for FPS calculation.
        """
        now = time.perf_counter()
        dt  = now - self._last_tick
        self._last_tick = now
        if dt > 0:
            self._frame_times.append(dt)

    def set_latency(self, latency_ms: float):
        """Manually set the processing latency for this frame."""
        self._latency_ms = latency_ms

    def get_stats(self) -> dict:
        """
        Return a dict of current performance statistics.

        Returns:
            {
                'fps'        : float — rolling average FPS
                'latency_ms' : float — last frame latency
                'cpu_pct'    : float — CPU usage percent (0 if psutil unavailable)
                'mem_mb'     : float — process memory in MB (0 if unavailable)
            }
        """
        fps = 0.0
        if self._frame_times:
            avg_dt = sum(self._frame_times) / len(self._frame_times)
            fps    = 1.0 / avg_dt if avg_dt > 0 else 0.0

        # Rate-limit system calls to every 1 second
        now = time.time()
        if _PSUTIL_AVAILABLE and (now - self._last_system_check) > 1.0:
            try:
                self._cached_cpu = psutil.cpu_percent(interval=None)
                self._cached_mem = self._process.memory_info().rss / (1024 * 1024)
                self._last_system_check = now
            except Exception:
                pass

        return {
            "fps":        fps,
            "latency_ms": self._latency_ms,
            "cpu_pct":    self._cached_cpu,
            "mem_mb":     self._cached_mem,
        }
