"""
AeroScript JARVIS — Text-to-Speech Service
===========================================
Thread-safe TTS wrapper around pyttsx3.

Design:
  - pyttsx3 must run on the same thread it was initialized on.
  - A dedicated daemon thread owns the engine and drains a speech queue.
  - Cooldown guard prevents the same phrase from repeating within N seconds.
  - TTS failures are logged and silently swallowed — never crash AeroScript.
"""

import queue
import threading
import time
import logging
import subprocess
import platform
from typing import Optional

log = logging.getLogger(__name__)
_perf_log = logging.getLogger("JARVIS.latency")

_SENTINEL = object()  # poison pill to stop the TTS thread


class TextToSpeechService:
    """
    Asynchronous TTS service using pyttsx3.

    Usage:
        tts = TextToSpeechService(rate=175, volume=0.9)
        tts.start()
        tts.speak("Canvas cleared.")
        tts.stop()
    """

    def __init__(
        self,
        rate: int = 175,
        volume: float = 0.90,
        cooldown_s: float = 3.0,
    ):
        self._rate     = rate
        self._volume   = volume
        self._cooldown = cooldown_s

        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._available = False
        self._use_fallback = False

        # Cooldown tracking: phrase → last_spoken_time
        self._last_spoken: dict[str, float] = {}
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    def start(self) -> bool:
        """
        Start the TTS daemon thread. Returns True if TTS is available.
        Safe to call even if pyttsx3 is broken — returns False instead.
        """
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-tts",
            daemon=True,
        )
        self._thread.start()
        # Wait up to 2.0s for the engine to initialize
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if self._available:
                break
            time.sleep(0.05)   # Optimized: 0.1→0.05 for faster init confirmation
        return self._available

    def stop(self):
        """Signal the TTS thread to exit cleanly."""
        if self._thread and self._thread.is_alive():
            self._queue.put(_SENTINEL)
            self._thread.join(timeout=3.0)

    def speak(self, text: str, force: bool = False):
        """
        Queue text for speaking.

        Args:
            text:  The string to speak.
            force: If True, bypass the cooldown guard.
        """
        if not self._available:
            log.debug(f"TTS unavailable — skipping: '{text}'")
            return

        now = time.time()
        with self._lock:
            last = self._last_spoken.get(text, 0.0)
            if not force and (now - last) < self._cooldown:
                log.debug(f"TTS cooldown active for: '{text}'")
                return
            self._last_spoken[text] = now

        # Clear any pending speech to avoid a backlog
        self._drain_queue()
        self._queue.put(text)

    def speak_force(self, text: str):
        """Speak immediately, bypassing cooldown."""
        self.speak(text, force=True)

    # ── Thread ────────────────────────────────────────────────────────────────

    def _run(self):
        """TTS thread body — owns the pyttsx3 engine."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            # Prefer a slightly more "robotic" voice if available
            voices = engine.getProperty("voices")
            if voices:
                # Try to find a male voice for JARVIS feel
                male = [v for v in voices if "male" in v.name.lower() or "david" in v.name.lower()]
                if male:
                    engine.setProperty("voice", male[0].id)
            self._available = True
            log.info("TTS engine initialized.")
        except Exception as e:
            log.warning(f"TTS engine failed to initialize: {e}")
            if platform.system() == "Darwin":
                # macOS: try the built-in `say` command as fallback
                # `say` exists on every Mac; test by asking it to say nothing
                try:
                    subprocess.run(["say", ""], capture_output=True, timeout=3)
                    self._use_fallback = True
                    self._available    = True
                    log.info("TTS falling back to macOS 'say' command — OK.")
                except Exception as e2:
                    log.warning(f"macOS 'say' unavailable: {e2}")
                    self._available = False
            else:
                self._available = False
            if not self._available:
                return

        while True:
            try:
                item = self._queue.get(timeout=0.05)  # Optimized: 0.5→0.05 for near-instant TTS response
                if item is _SENTINEL:
                    break
                
                tts_start = time.perf_counter()
                if self._use_fallback:
                    subprocess.run(["say", item], check=False)
                else:
                    engine.say(item)
                    engine.runAndWait()
                
                tts_dur = (time.perf_counter() - tts_start) * 1000
                _perf_log.info(f"TTS speak: \"{item[:40]}\" took {tts_dur:.0f}ms")
                self._queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.error(f"TTS speak error: {e}")

        if not self._use_fallback:
            try:
                engine.stop()
            except Exception:
                pass
        log.info("TTS thread stopped.")

    def _drain_queue(self):
        """Discard any pending (not-yet-spoken) items in the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
