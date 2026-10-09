"""
AeroScript JARVIS — Speech-to-Text Service
===========================================
Background speech recognition using the `speech_recognition` library.

Modes:
  A) Push-to-Talk  — caller invokes listen_once() / stop_listening()
  B) Background    — continuous listen_background() with callback

The service NEVER blocks the OpenCV/MediaPipe loop.
All results are delivered via callback on the STT daemon thread;
the caller is responsible for queuing results safely.

Error hierarchy handled:
  - No microphone / permission denied → logs + sets available=False
  - Timeout                           → callback(None)
  - Unrecognised speech               → callback(None)
  - Network unavailable               → callback(None) + logs warning
"""

import re
import threading
import logging
import time
import os
import numpy as np
from typing import Optional, Callable
from contextlib import contextmanager

from config import settings
from .latency_tracker import LatencyTracker

# ── Domain Keyword & Phonetic Normalization ──────────────────────────────────
DOMAIN_KEYWORDS = [
    "gemini", "open gemini", "close gemini", "ask gemini", "search gemini",
    "whiteboard", "camera", "canvas", "clear canvas", "clear",
    "undo", "redo", "screenshot", "save canvas", "eraser",
    "highlighter", "marker", "pen", "smart math", "air text",
    "presentation", "annotation", "pause", "resume",
]

def correct_domain_phrases(text: str) -> str:
    """
    Correct common speech recognition mishearings for AeroScript voice commands,
    specifically focusing on Gemini and common app terms where cloud STT
    frequently predicts 'Gmail', 'gem', or 'germany'.
    """
    if not text:
        return text

    raw = text.strip()
    low = raw.lower()

    # Open Gemini variations
    gemini_open_patterns = [
        r"\bopen\s+(?:gmail|gem|germany|jamini|jemini|jini|jimmy|g\s*mini|gem\s*in\s*eye|gem\s*eye|gemma)\b",
        r"\bstart\s+(?:gmail|gem|germany|jamini|jemini|jini|g\s*mini)\b",
        r"\blaunch\s+(?:gmail|gem|germany|jamini|jemini|jini|g\s*mini)\b",
        r"\benable\s+(?:gmail|gem|germany|jamini|jemini|jini|g\s*mini)\b",
    ]
    for pat in gemini_open_patterns:
        if re.search(pat, low):
            return re.sub(pat, "open gemini", raw, flags=re.IGNORECASE)

    # Close Gemini variations
    gemini_close_patterns = [
        r"\bclose\s+(?:gmail|gem|germany|jamini|jemini|jini|jimmy|g\s*mini)\b",
        r"\bexit\s+(?:gmail|gem|germany|jamini|jemini|jini|g\s*mini)\b",
        r"\bstop\s+(?:gmail|gem|germany|jamini|jemini|jini|g\s*mini)\b",
    ]
    for pat in gemini_close_patterns:
        if re.search(pat, low):
            return re.sub(pat, "close gemini", raw, flags=re.IGNORECASE)

    # Ask / Search Gemini variations
    gemini_ask_patterns = [
        r"\bask\s+(?:gmail|gem|germany|jamini|jemini|jini)\b",
        r"\bsearch\s+(?:in|on)?\s*(?:gmail|gem|germany|jamini|jemini|jini)\b",
    ]
    for pat in gemini_ask_patterns:
        if re.search(pat, low):
            return re.sub(pat, "ask gemini", raw, flags=re.IGNORECASE)

    # Standalone single words that are misheard for 'gemini'
    if low in ("gem", "gmail", "jamini", "jemini", "jini", "g mini", "gem in eye"):
        return "gemini"

    # In phrases, replace common phonetic variants
    sub_patterns = [
        (r"\bg\s*mini\b", "gemini"),
        (r"\bjamini\b", "gemini"),
        (r"\bjemini\b", "gemini"),
        (r"\bgem\s+in\s+eye\b", "gemini"),
    ]
    result = raw
    for pat, repl in sub_patterns:
        result = re.sub(pat, repl, result, flags=re.IGNORECASE)

    return result

# ── macOS SSL fix ─────────────────────────────────────────────────────────────
# Python on macOS doesn't use system SSL certs by default.
# This causes Google STT API calls to silently fail with CERTIFICATE_VERIFY_FAILED.
# Fix: point SSL to the certifi bundle before any network calls.
try:
    import certifi
    import ssl
    os.environ["SSL_CERT_FILE"]      = certifi.where()
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    pass

log = logging.getLogger(__name__)


class SpeechToTextService:
    """
    Non-blocking speech recognition wrapper.

    Usage:
        def on_text(text: str | None):
            if text:
                queue.put(text)

        stt = SpeechToTextService(on_text)
        stt.start()

        # Push-to-talk:
        stt.begin_listening()   # when V pressed
        stt.stop_listening()    # when V pressed again (or auto after timeout)

        stt.shutdown()
    """

    def __init__(
        self,
        callback: Callable[[Optional[str]], None],
        energy_threshold: int = 300,
        dynamic_energy: bool = True,
        listen_timeout: float = 6.0,
        phrase_timeout: float = 5.0,
        language: Optional[str] = None,
    ):
        self._callback       = callback
        self._energy         = energy_threshold
        self._dynamic_energy = dynamic_energy
        self._listen_timeout = listen_timeout
        self._phrase_timeout = phrase_timeout
        self._language       = language or getattr(settings, "JARVIS_STT_LANGUAGE", "en-US")

        self._available      = False
        self._listening      = False   # True while actively recording
        self._shutdown_flag  = threading.Event()

        self._listen_thread: Optional[threading.Thread] = None
        self._mic_idx        = None
        self._recognizer     = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available

    @property
    def listening(self) -> bool:
        return self._listening

    def _get_working_mic(self):
        """
        Return an sr.Microphone instance guaranteed to open a valid non-None audio stream.
        Tries self._mic_idx first, then default None, then candidate indices.
        """
        import speech_recognition as sr
        candidates = []
        if self._mic_idx is not None:
            candidates.append(self._mic_idx)
        candidates.append(None)  # system default mic

        try:
            working = sr.Microphone.list_working_microphones()
            for idx in working:
                if idx not in candidates:
                    candidates.append(idx)
        except Exception:
            pass

        for idx in candidates:
            try:
                mic = sr.Microphone(device_index=idx) if idx is not None else sr.Microphone()
                src = mic.__enter__()
                if getattr(mic, 'stream', None) is None:
                    raise OSError(f"Device index {idx} stream is None")
                try:
                    mic.__exit__(None, None, None)
                except Exception:
                    pass
                self._mic_idx = idx
                return mic
            except Exception as e:
                log.debug(f"Microphone index {idx} failed validation: {e}")
                continue

        return sr.Microphone()

    @contextmanager
    def _open_mic_stream(self):
        """Safe context manager to enter/exit sr.Microphone without NoneType stream errors."""
        mic = self._get_working_mic()
        entered = False
        try:
            src = mic.__enter__()
            entered = True
            if getattr(mic, 'stream', None) is None:
                raise OSError("Opened microphone stream is None")
            yield src
        finally:
            if entered and getattr(mic, 'stream', None) is not None:
                try:
                    mic.__exit__(None, None, None)
                except Exception as ex:
                    log.warning(f"Error closing microphone stream: {ex}")

    def _get_mic(self):
        import speech_recognition as sr
        if self._mic_idx is not None:
            return sr.Microphone(device_index=self._mic_idx)
        return sr.Microphone()

    def start(self) -> bool:
        """
        Initialize recognizer and microphone. Returns True if ready.
        Safe to call even when no microphone is present.
        """
        try:
            import speech_recognition as sr
            self._recognizer = sr.Recognizer()
            self._recognizer.energy_threshold         = self._energy
            self._recognizer.dynamic_energy_threshold = settings.JARVIS_STT_DYNAMIC_ENERGY
            self._recognizer.dynamic_energy_adjustment_damping = 0.15 # Default damping to prevent runaway drift
            self._recognizer.dynamic_energy_ratio     = 1.5
            self._recognizer.pause_threshold          = getattr(settings, "VOICE_PAUSE_THRESHOLD", 0.8) # 0.8s allows natural syllable pauses without truncating words like 'gemini'
            self._recognizer.phrase_threshold         = 0.3    # Firmer speech start (ignores keyboard clicks)
            self._recognizer.non_speaking_duration    = 0.5    # Padding to ensure phrase endings aren't clipped

            # Select physical built-in / USB / headset microphone, strictly avoiding sleeping iPhone / continuity
            working_mics = {}
            try:
                working_mics = sr.Microphone.list_working_microphones()
            except Exception:
                pass

            mics = sr.Microphone.list_microphone_names()
            self._mic_idx = None

            # Priority 1: Headset / AirPods / USB microphone in working mics
            for i, name in enumerate(mics):
                if working_mics and i not in working_mics:
                    continue
                nl = name.lower()
                if any(k in nl for k in ["iphone", "continuity", "teams", "zoom", "speaker", "output", "virtual"]):
                    continue
                if any(k in nl for k in ["airpods", "headset", "headphone", "usb", "external"]):
                    self._mic_idx = i
                    break

            # Priority 2: Built-in MacBook / internal microphone in working mics
            if self._mic_idx is None:
                for i, name in enumerate(mics):
                    if working_mics and i not in working_mics:
                        continue
                    nl = name.lower()
                    if any(k in nl for k in ["iphone", "continuity", "teams", "zoom", "speaker", "output", "virtual"]):
                        continue
                    if any(k in nl for k in ["macbook", "built-in", "internal", "primary"]):
                        self._mic_idx = i
                        break

            # Priority 3: First valid working physical device
            if self._mic_idx is None:
                for i, name in enumerate(mics):
                    if working_mics and i not in working_mics:
                        continue
                    nl = name.lower()
                    if any(k in nl for k in ["iphone", "continuity", "teams", "zoom", "speaker", "output", "virtual"]):
                        continue
                    self._mic_idx = i
                    break

            mic_name = mics[self._mic_idx] if self._mic_idx is not None else "system default"
            log.info(f"Using microphone: index {self._mic_idx} ({mic_name})")

            # Verify microphone is accessible and calibrate ambient noise ONCE at startup
            try:
                with self._open_mic_stream() as src:
                    log.info("Calibrating ambient noise (0.8s)...")
                    self._recognizer.adjust_for_ambient_noise(src, duration=0.8)
                    # If dynamic is enabled, we still clamp the initial calibration
                    self._recognizer.energy_threshold = max(70, min(self._recognizer.energy_threshold * 1.2, 180))
                    log.info(f"Microphone ready: initial threshold={self._recognizer.energy_threshold:.1f}")
            except Exception as e:
                log.warning(f"Ambient calibration note: {e}, using default threshold {self._energy}")
                self._recognizer.energy_threshold = self._energy

            self._available = True
            log.info("STT service ready.")
            return True

        except OSError as e:
            log.warning(f"Microphone unavailable: {e}")
            self._available = False
            return False
        except Exception as e:
            log.warning(f"STT init failed: {e}")
            self._available = False
            return False

    def shutdown(self):
        """Stop all listening threads cleanly."""
        self._shutdown_flag.set()
        self._listening = False
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=4.0)

    def begin_listening(self):
        """
        Start a one-shot listen cycle on a background thread.
        The result (str or None) is delivered to the callback.
        """
        if not self._available:
            log.debug("STT unavailable — begin_listening() ignored.")
            return
        if self._listening:
            log.debug("Already listening.")
            return

        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_once,
            name="jarvis-stt-ptt",
            daemon=True,
        )
        self._listen_thread.start()

    def stop_listening(self):
        """Signal that push-to-talk was released (does not abort mid-phrase)."""
        self._listening = False

    # ── Background continuous listening (Always Listening mode) ─────────────

    def start_continuous(self) -> bool:
        """
        Start permanent continuous background listening.
        Microphone stream is held open, and recognized speech is delivered to callback.
        """
        if not self._available:
            return False
        if self._listening and self._listen_thread and self._listen_thread.is_alive():
            return True

        self._shutdown_flag.clear()
        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._continuous_loop,
            name="jarvis-stt-continuous",
            daemon=True,
        )
        self._listen_thread.start()
        log.info("Continuous speech listening thread started.")
        return True

    def start_background(self) -> bool:
        """Alias for start_continuous."""
        return self.start_continuous()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_audio(self, audio):
        """Recognize an audio snippet in a worker thread and deliver callback."""
        tracker = LatencyTracker()
        tracker.stamp("speech_captured")
        try:
            raw = audio.get_raw_data()
            # Fast numpy-based silence check (replaces slow Python byte iteration)
            raw_np = np.frombuffer(raw[:2048], dtype=np.uint8)
            if not np.any(raw_np):
                log.warning("Captured audio is completely silent (all 0s) - check macOS Microphone privacy permission.")
                self._callback("__MIC_MUTED_OR_DENIED__")
                return

            # Software Noise Gate: check RMS energy to drop pure static
            rms = np.sqrt(np.mean(np.square(raw_np.astype(np.float32) - 128.0)))
            if rms < 5.0: # Very low energy, just static
                log.debug(f"Audio rejected by software noise gate (RMS: {rms:.1f})")
                return

            text = self._recognize(audio, tracker)
            if text and text not in ("__SPEECH_UNRECOGNIZED__", None):
                self._callback(text, tracker)
        except Exception as e:
            log.error(f"Error processing captured audio: {e}", exc_info=True)

    def _listen_once(self):
        """Record a single audio phrase and recognize it."""
        import speech_recognition as sr
        tracker = LatencyTracker()
        tracker.stamp("speech_captured")
        try:
            with self._open_mic_stream() as src:
                # Do NOT recalibrate ambient noise here — doing so eats the start of speech!
                log.info(f"Listening for speech (threshold={self._recognizer.energy_threshold:.1f})...")
                audio = self._recognizer.listen(
                    src,
                    timeout=self._listen_timeout,
                    phrase_time_limit=self._phrase_timeout,
                )

            raw = audio.get_raw_data()
            # Fast numpy-based silence check
            raw_np = np.frombuffer(raw[:2048], dtype=np.uint8)
            if not np.any(raw_np):
                log.warning("Captured audio is completely silent (all 0s) - check macOS Microphone privacy permission.")
                self._callback("__MIC_MUTED_OR_DENIED__")
                return

            # Software Noise Gate: check RMS energy to drop pure static
            rms = np.sqrt(np.mean(np.square(raw_np.astype(np.float32) - 128.0)))
            if rms < 5.0: # Very low energy, just static
                log.debug(f"Audio rejected by software noise gate (RMS: {rms:.1f})")
                self._callback(None)
                return

            log.info(f"Audio captured ({len(raw)} bytes), sending to speech recognition...")
            text = self._recognize(audio, tracker)
            self._callback(text, tracker)

        except sr.WaitTimeoutError:
            log.info("Listen timeout — no speech detected within timeout.")
            self._callback(None)
        except OSError as e:
            log.warning(f"Microphone error during listen: {e}")
            self._callback(None)
        except Exception as e:
            log.error(f"Unexpected STT error: {e}", exc_info=True)
            self._callback(None)
        finally:
            self._listening = False

    def _continuous_loop(self):
        """Continuous background listening loop. Holds microphone open permanently."""
        import speech_recognition as sr
        log.info("Continuous STT loop active.")
        while not self._shutdown_flag.is_set():
            try:
                with self._open_mic_stream() as src:
                    log.info("Microphone stream opened for continuous listening.")
                    while not self._shutdown_flag.is_set():
                        try:
                            audio = self._recognizer.listen(
                                src,
                                timeout=1.5,
                                phrase_time_limit=self._phrase_timeout,
                            )
                            # Spawn recognition worker so listening never pauses or drops words
                            threading.Thread(
                                target=self._process_audio,
                                args=(audio,),
                                daemon=True,
                                name="jarvis-stt-worker",
                            ).start()
                        except sr.WaitTimeoutError:
                            continue
                        except Exception as e:
                            log.debug(f"Audio tick note: {e}")
            except OSError as e:
                log.warning(f"Microphone device error in continuous loop: {e}")
                time.sleep(1.5)
            except Exception as e:
                log.error(f"Unexpected continuous STT error: {e}", exc_info=True)
                time.sleep(1.0)
        self._listening = False
        log.info("Continuous STT loop stopped.")

    def _background_loop(self):
        """Fallback alias for continuous loop."""
        self._continuous_loop()

    def _recognize(self, audio, tracker: Optional[LatencyTracker] = None) -> Optional[str]:
        """Run Google STT on the captured audio. Returns text or status marker."""
        import speech_recognition as sr
        try:
            # Request all alternatives to parse confidence scores & candidate phrases
            results = self._recognizer.recognize_google(
                audio,
                language=self._language,
                show_all=True
            )
            if not results or not isinstance(results, dict) or 'alternative' not in results:
                raise sr.UnknownValueError()

            alternatives = results.get('alternative', [])
            if not alternatives:
                raise sr.UnknownValueError()

            # Smart alternative selection:
            # Prioritize candidates that contain application domain keywords (e.g. "gemini")
            best_alt = alternatives[0]
            for alt in alternatives:
                cand_text = alt.get('transcript', '').strip().lower()
                if any(kw in cand_text for kw in ("gemini", "whiteboard", "canvas", "screenshot", "smart math", "air text", "clear")):
                    best_alt = alt
                    break

            raw_text = best_alt.get('transcript', '')
            confidence = best_alt.get('confidence', 1.0)

            if tracker:
                tracker.stamp("stt_complete")
            stt_ms = tracker.delta_ms("speech_captured", "stt_complete") if tracker else 0

            # Apply domain phonetic correction (e.g. 'open gmail' / 'open gem' -> 'open gemini')
            text = correct_domain_phrases(raw_text)

            if confidence < 0.60 and text == raw_text:
                log.warning(f"Speech rejected due to low confidence ({confidence:.2f}): '{text}'")
                return "__SPEECH_UNRECOGNIZED__"

            if text != raw_text:
                log.info(f"Recognized (corrected): '{text}' [Original: '{raw_text}'] (Conf: {confidence:.2f}, STT: {stt_ms:.0f}ms)")
            else:
                log.info(f"Recognized: '{text}' (Conf: {confidence:.2f}, STT: {stt_ms:.0f}ms)")
            return text
        except sr.UnknownValueError:
            log.debug("Speech not understood.")
            return "__SPEECH_UNRECOGNIZED__"
        except sr.RequestError as e:
            log.warning(f"Google STT network error: {e}")
            return "__NETWORK_ERROR__"
        except Exception as e:
            log.error(f"Recognition error: {e}")
            return None
