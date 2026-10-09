"""
AeroScript JARVIS — Wake-Word Detector
=======================================
Lightweight wake-word detection using pattern matching on short audio
snippets. No heavy dependencies (no Porcupine, no neural nets).

Strategy:
  - A background thread continuously listens for short snippets (2.5s).
  - If recognized text contains a configured wake word ("wake up", "hey jarvis", etc.):
      - Extracts any command spoken in the same breath (e.g. "wake up open gemini").
      - Pauses itself to yield microphone access to SpeechToTextService.
      - Fires callback to trigger acknowledgement and command listening.
  - Automatically paused during command recording and resumed once done.

Accuracy note:
  Pattern matching on Google STT output is fast and zero-install.
  Enabled/disabled via config: JARVIS_WAKE_WORD_ENABLED
"""

import re
import threading
import time
import logging
from typing import Callable, Optional

log = logging.getLogger(__name__)

DEFAULT_WAKE_WORDS = [
    "hey jarvis", "ok jarvis", "okay jarvis", "jarvis",
    "wake up", "wake up jarvis", "wake up aeroscript",
    "hi jarvis", "hello jarvis", "aero script", "aeroscript",
]


class WakeWordDetector:
    """
    Wake-word detector supporting 'wake up', 'hey jarvis', etc.

    When wake words are detected in short audio snippets, calls
    `on_wake_word(command_text)`. The caller then starts full-phrase listening
    or executes the inline command immediately.

    Usage:
        def on_wake(cmd=None):
            jarvis.begin_command_listening(cmd)

        wwd = WakeWordDetector(
            on_wake_word=on_wake,
            wake_words=["wake up", "hey jarvis", "jarvis"],
        )
        wwd.start()
        ...
        wwd.stop()
    """

    def __init__(
        self,
        on_wake_word: Callable[[Optional[str]], None],
        wake_words: Optional[list[str]] = None,
        energy_threshold: int = 300,
        dynamic_energy: bool = True,
        snippet_duration: float = 2.5,
        mic_idx: Optional[int] = None,
    ):
        self._on_wake        = on_wake_word
        self._wake_words     = [w.lower().strip() for w in (wake_words or DEFAULT_WAKE_WORDS)]
        self._energy         = energy_threshold
        self._dynamic_energy = dynamic_energy
        self._snippet_dur    = snippet_duration
        self._mic_idx        = mic_idx

        self._thread: Optional[threading.Thread] = None
        self._stop_event     = threading.Event()
        self._paused         = threading.Event()
        self._available      = False

        # Cooldown so one detection doesn't fire repeatedly
        self._last_fired     = 0.0
        self._cooldown_s     = 2.0

        # Safety resume: auto-resume after this timeout (seconds) to prevent permanent pause
        self._auto_resume_timeout = 15.0
        self._pause_time     = 0.0
        self._resume_timer: Optional[threading.Timer] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def is_paused(self) -> bool:
        return self._paused.is_set()

    def set_mic_index(self, idx: Optional[int]):
        """Bind to the working microphone device index selected by STT."""
        self._mic_idx = idx

    def pause(self):
        """Temporarily pause detection to yield microphone to STT recording."""
        self._paused.set()
        self._pause_time = time.time()
        # Schedule a safety auto-resume in case STT never calls resume()
        self._cancel_resume_timer()
        self._resume_timer = threading.Timer(
            self._auto_resume_timeout, self._safety_resume
        )
        self._resume_timer.daemon = True
        self._resume_timer.start()
        log.debug(f"Wake-word detector paused (auto-resume in {self._auto_resume_timeout}s)")

    def resume(self):
        """Resume detection when STT recording is finished."""
        self._cancel_resume_timer()
        if self._paused.is_set():
            elapsed = time.time() - self._pause_time if self._pause_time else 0
            log.debug(f"Wake-word detector resumed (was paused {elapsed:.1f}s)")
        self._paused.clear()

    def _cancel_resume_timer(self):
        """Cancel any pending safety resume timer."""
        if self._resume_timer and self._resume_timer.is_alive():
            self._resume_timer.cancel()
        self._resume_timer = None

    def _safety_resume(self):
        """Auto-resume after timeout — prevents permanently paused state."""
        if self._paused.is_set() and not self._stop_event.is_set():
            log.warning(f"Wake-word detector auto-resuming after {self._auto_resume_timeout}s safety timeout.")
            self._paused.clear()

    def _get_mic(self):
        import speech_recognition as sr
        if self._mic_idx is not None:
            try:
                return sr.Microphone(device_index=self._mic_idx)
            except Exception:
                pass
        return sr.Microphone()

    def start(self) -> bool:
        """Start wake-word detection in a background thread."""
        try:
            import speech_recognition as sr
            import pyaudio  # noqa: F401 — verify audio hardware available
            self._available = True
        except ImportError as e:
            log.warning(f"Wake-word detector unavailable: {e}")
            return False

        self._stop_event.clear()
        self._paused.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-wakeword",
            daemon=True,
        )
        self._thread.start()
        log.info(f"Wake-word detector started. Listening for: {self._wake_words}")
        return True

    def stop(self):
        """Stop the wake-word detection thread."""
        self._stop_event.set()
        self._paused.clear()
        self._cancel_resume_timer()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=4.0)
        log.info("Wake-word detector stopped.")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _extract_wake_word(self, text: str) -> tuple[bool, Optional[str]]:
        """
        Check if text contains any configured wake word.
        Returns (matched, remainder_command).
        """
        if not text:
            return False, None

        t = text.lower().strip()
        sorted_wws = sorted(self._wake_words, key=len, reverse=True)

        for ww in sorted_wws:
            # Matches "wake up <command>" or "<wake word>, <command>"
            pattern = r"^\b" + re.escape(ww) + r"\b(?:\s*[,:]?\s*(.*))?$"
            m = re.match(pattern, t)
            if m:
                remainder = m.group(1).strip() if m.group(1) else None
                return True, remainder

            # Matches wake word anywhere in the utterance
            if re.search(r"\b" + re.escape(ww) + r"\b", t):
                sub_pattern = r"\b" + re.escape(ww) + r"\b(?:\s*[,:]?\s*(.*))?$"
                sm = re.search(sub_pattern, t)
                remainder = sm.group(1).strip() if (sm and sm.group(1)) else None
                return True, remainder

        return False, None

    def _run(self):
        """Background loop — listens for short snippets, checks for wake words."""
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        recognizer.energy_threshold         = self._energy
        recognizer.dynamic_energy_threshold = self._dynamic_energy
        recognizer.pause_threshold          = 0.5
        recognizer.phrase_threshold         = 0.25

        log.debug("Wake-word loop running.")
        while not self._stop_event.is_set():
            if self._paused.is_set():
                time.sleep(0.15)
                continue

            try:
                with self._get_mic() as src:
                    while not self._stop_event.is_set() and not self._paused.is_set():
                        try:
                            audio = recognizer.listen(
                                src,
                                timeout=1.5,
                                phrase_time_limit=self._snippet_dur,
                            )
                        except sr.WaitTimeoutError:
                            continue

                        if self._paused.is_set() or self._stop_event.is_set():
                            break

                        # Recognize off the hot path
                        text = self._recognize_snippet(recognizer, audio)
                        if text:
                            matched, remainder = self._extract_wake_word(text)
                            if matched:
                                now = time.time()
                                if now - self._last_fired > self._cooldown_s:
                                    self._last_fired = now
                                    log.info(f"Wake word detected in: '{text}' (remainder: '{remainder}')")
                                    self.pause()  # Yield mic immediately
                                    try:
                                        self._on_wake(remainder)
                                    except TypeError:
                                        try:
                                            self._on_wake()
                                        except Exception as e:
                                            log.error(f"Wake callback error (no args): {e}")
                                            self.resume()  # Don't leave detector stuck
                                    except Exception as e:
                                        log.error(f"Wake callback error: {e}")
                                        self.resume()  # Don't leave detector stuck
                                    break
            except OSError as e:
                log.warning(f"Mic error in wake-word loop: {e}")
                time.sleep(1.5)
            except Exception as e:
                log.error(f"Wake-word loop error: {e}")
                time.sleep(1.0)

        log.debug("Wake-word loop exited.")

    def _recognize_snippet(self, recognizer, audio) -> Optional[str]:
        """Quick recognition of a short audio clip. Returns lowercase text or None."""
        import speech_recognition as sr
        from config import settings
        lang = getattr(settings, "JARVIS_STT_LANGUAGE", "en-US")
        try:
            text = recognizer.recognize_google(audio, language=lang)
            return text.lower()
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
        except Exception:
            return None
