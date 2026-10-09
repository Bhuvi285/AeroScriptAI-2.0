"""
AeroScript JARVIS — Orchestrator
==================================
Top-level voice assistant that owns all sub-services:
  - SpeechToTextService   (background thread)
  - TextToSpeechService   (background thread)
  - WakeWordDetector      (optional background thread)
  - CommandProcessor      (stateless, called on STT thread)

The ONLY thread-safe interface to the main OpenCV loop is via:
  jarvis.drain_commands() → list[VoiceCommand]

Everything else runs in daemon threads.

State machine:
  OFFLINE → READY → LISTENING → PROCESSING → SPEAKING → READY
                                                        ↓
                                                      ERROR → READY
"""

import queue
import threading
import time
import logging
import subprocess
from enum import Enum, auto
from typing import Optional

from .speech_to_text  import SpeechToTextService
from .text_to_speech  import TextToSpeechService
from .wake_word       import WakeWordDetector
from .command_processor import CommandProcessor, VoiceCommand, Intent
from .latency_tracker import LatencyTracker
from .responses       import (
    get_response, get_toast,
    STARTUP_GREETING, STARTUP_GREETING_SIMPLE, WAKE_ACK,
    NO_MICROPHONE, NETWORK_ERROR, TTS_ERROR,
)

log = logging.getLogger(__name__)


# ── JARVIS Status ─────────────────────────────────────────────────────────────

class JarvisStatus(Enum):
    OFFLINE    = auto()
    READY      = auto()
    LISTENING  = auto()
    PROCESSING = auto()
    SPEAKING   = auto()
    ERROR      = auto()


# UI colours for each status (BGR for OpenCV)
STATUS_COLORS: dict[JarvisStatus, tuple[int, int, int]] = {
    JarvisStatus.OFFLINE:    (80,  80,  80),   # grey
    JarvisStatus.READY:      (50,  200, 50),   # green
    JarvisStatus.LISTENING:  (0,   220, 255),  # yellow / cyan
    JarvisStatus.PROCESSING: (200, 140, 0),   # blue
    JarvisStatus.SPEAKING:   (180, 50,  220),  # purple
    JarvisStatus.ERROR:      (40,  40,  220),  # red
}

STATUS_LABELS: dict[JarvisStatus, str] = {
    JarvisStatus.OFFLINE:    "OFFLINE",
    JarvisStatus.READY:      "READY",
    JarvisStatus.LISTENING:  "LISTENING",
    JarvisStatus.PROCESSING: "PROCESSING",
    JarvisStatus.SPEAKING:   "SPEAKING",
    JarvisStatus.ERROR:      "ERROR",
}


# ── Orchestrator ──────────────────────────────────────────────────────────────

class JarvisAssistant:
    """
    High-level JARVIS voice assistant.

    Thread safety:
        - All public methods except drain_commands() are thread-safe.
        - drain_commands() must be called ONLY from the main OpenCV thread.
        - SpeechToTextService delivers on its own daemon thread.
        - TextToSpeechService speaks on its own daemon thread.

    Lifecycle:
        # App init
        self.jarvis = JarvisAssistant(startup_speech=True, wake_word=False)
        ok = self.jarvis.start()

        # Each frame (main OpenCV loop)
        cmds = self.jarvis.drain_commands()
        for cmd in cmds:
            self.controller.execute(cmd)

        # V key pressed
        self.jarvis.toggle_listening()

        # Cleanup
        self.jarvis.stop()
    """

    def __init__(
        self,
        startup_speech:   bool = True,
        wake_word_enabled: bool = False,
        wake_words:        Optional[list[str]] = None,
        energy_threshold:  int  = 300,
        listen_timeout:    float = 5.0,
        phrase_timeout:    float = 3.0,
        tts_rate:          int  = 175,
        tts_volume:        float = 0.90,
        response_cooldown: float = 3.0,
    ):
        self._startup_speech    = startup_speech
        self._wake_word_enabled = wake_word_enabled
        self._wake_words        = wake_words or [
            "hey jarvis", "ok jarvis", "okay jarvis", "jarvis",
            "wake up", "wake up jarvis", "wake up aeroscript",
            "hi jarvis", "hello jarvis", "aero script", "aeroscript",
        ]

        # Command queue — STT thread puts, main thread drains
        self._cmd_queue: queue.Queue[VoiceCommand] = queue.Queue()

        # Status (read by UI thread — intentionally not locked; worst case stale for 1 frame)
        self._status = JarvisStatus.OFFLINE

        # Last recognized text (for UI display)
        self._last_text   = ""
        self._last_text_t = 0.0

        # Sub-services
        self._stt = SpeechToTextService(
            callback=self._on_speech,
            energy_threshold=energy_threshold,
            dynamic_energy=True,
            listen_timeout=listen_timeout,
            phrase_timeout=phrase_timeout,
        )
        self._tts = TextToSpeechService(
            rate=tts_rate,
            volume=tts_volume,
            cooldown_s=response_cooldown,
        )
        self._wwd: Optional[WakeWordDetector] = None
        if wake_word_enabled:
            self._wwd = WakeWordDetector(
                on_wake_word=self._on_wake_word,
                wake_words=self._wake_words,
                energy_threshold=energy_threshold,
            )

        self._processor = CommandProcessor()

        # Push-to-talk toggle state
        self._ptt_active = False
        self._app        = None

    def bind_app(self, app):
        """Bind App instance for direct UI feedback (toasts)."""
        self._app = app

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def status(self) -> JarvisStatus:
        return self._status

    @property
    def last_text(self) -> str:
        return self._last_text

    @property
    def stt_available(self) -> bool:
        return self._stt.available

    @property
    def tts_available(self) -> bool:
        return self._tts.available

    def start(self) -> bool:
        """
        Initialize all services. Returns True if at least STT is available.
        Gracefully handles partial failures (e.g. TTS broken → STT still works).
        """
        log.info("JARVIS starting...")

        # TTS first (no mic needed)
        tts_ok = self._tts.start()
        if not tts_ok:
            log.warning(TTS_ERROR)

        # STT
        stt_ok = self._stt.start()
        if not stt_ok:
            log.warning(NO_MICROPHONE)
            self._status = JarvisStatus.OFFLINE
            return False

        # Initial state should be READY, not continuously listening
        if stt_ok:
            self._status = JarvisStatus.READY
            log.info("JARVIS: Microphone initialized. (Click or press V to talk)")

        # Optional wake word
        if self._wwd:
            if hasattr(self._stt, "_mic_idx"):
                self._wwd.set_mic_index(self._stt._mic_idx)
            self._wwd.start()

        log.info("JARVIS ready and listening.")

        # Startup greeting — re-check TTS (it may have finished init by now)
        if self._startup_speech and self._tts.available:
            self._tts.speak_force(STARTUP_GREETING_SIMPLE)

        return True

    def stop(self):
        """Clean shutdown of all services."""
        log.info("JARVIS shutting down...")
        self._status = JarvisStatus.OFFLINE
        self._stt.shutdown()
        self._tts.stop()
        if self._wwd:
            self._wwd.stop()
        log.info("JARVIS stopped.")

    def listen_once_async(self):
        """Trigger a single phrase listening cycle (e.g. from UI click or V key)."""
        if not self._stt.available:
            log.debug("Listen ignored — STT unavailable.")
            if self._app:
                self._app._toast("JARVIS: Microphone unavailable")
            return

        if self._status in (JarvisStatus.LISTENING, JarvisStatus.PROCESSING):
            return  # Already busy

        self._status = JarvisStatus.LISTENING
        self._stt.begin_listening()
        if self._app:
            self._app._toast("JARVIS: Listening...")

    def toggle_listening(self):
        """
        Triggered by 'V' key as push-to-talk fallback.
        In Phase 3, this triggers a single listening cycle rather than continuous mode.
        """
        self.listen_once_async()

    def drain_commands(self) -> list[VoiceCommand]:
        """
        Drain all pending VoiceCommands from the queue.
        Call ONCE per OpenCV frame from the main thread.

        Returns:
            List of VoiceCommand objects (may be empty).
        """
        cmds = []
        while not self._cmd_queue.empty():
            try:
                cmds.append(self._cmd_queue.get_nowait())
            except queue.Empty:
                break
        return cmds

    def speak(self, text: str):
        """Speak a response (non-blocking). Called from main thread after command execution."""
        if self._tts.available:
            self._status = JarvisStatus.SPEAKING
            self._tts.speak(text)
            # Immediately schedule status reset (no artificial 500ms delay)
            self._set_ready()

    def speak_force(self, text: str):
        """Speak immediately, bypassing cooldown."""
        if self._tts.available:
            self._tts.speak_force(text)

    # ── Internal Callbacks ────────────────────────────────────────────────────

    def _on_speech(self, text: Optional[str], tracker: Optional[LatencyTracker] = None):
        """
        Called by SpeechToTextService on the STT thread when a phrase is recognized.
        Must be thread-safe — only puts to the queue, never touches App state.
        """
        try:
            if text == "__MIC_MUTED_OR_DENIED__":
                self._status = JarvisStatus.READY
                if self._app:
                    self._app._toast("JARVIS: Mic blocked! Check macOS Privacy > Microphone")
                log.warning("Microphone stream was silent (all zeros) - check macOS microphone permission")
                return

            if text == "__SPEECH_UNRECOGNIZED__":
                self._status = JarvisStatus.READY
                return

            if text == "__NETWORK_ERROR__":
                self._status = JarvisStatus.READY
                if self._app:
                    self._app._toast("JARVIS: Speech recognition offline (check internet)")
                return

            if text is None:
                self._status = JarvisStatus.READY
                return

            # Create tracker if STT didn't provide one
            if tracker is None:
                tracker = LatencyTracker(text)
                tracker.stamp("speech_captured")
                tracker.stamp("stt_complete")
            else:
                tracker._command_text = text

            self._status = JarvisStatus.PROCESSING
            self._last_text   = text
            self._last_text_t = time.time()
            log.info(f"STT result: '{text}'")
            if self._app:
                self._app._toast(f"JARVIS heard: '{text}'")

            chat_active = bool(self._app and getattr(self._app, "chat_mode", False))
            tracker.stamp("intent_start")
            cmd = self._processor.process(text, chat_mode_active=chat_active)
            tracker.stamp("intent_detected")
            cmd._tracker = tracker
            tracker.stamp("command_queued")
            self._cmd_queue.put(cmd)
            self._status = JarvisStatus.READY
        finally:
            if self._wwd:
                self._wwd.resume()

    def _on_wake_word(self, command_text: Optional[str] = None):
        """
        Called by WakeWordDetector on its thread when a wake word is heard.
        If command_text is already present (e.g. 'wake up open gemini'), executes directly.
        Otherwise, acknowledges with WAKE_ACK and begins listening for the command.
        """
        log.info(f"Wake word detected. (command_text='{command_text}')")
        if self._wwd:
            self._wwd.pause()

        # If user spoke command in the same breath, execute immediately
        if command_text and command_text.strip():
            log.info(f"Executing command from wake phrase: '{command_text}'")
            if self._app:
                self._app._toast(f"JARVIS: '{command_text}'")
            self._on_speech(command_text)
            return

        # Acknowledge and start listening for the user's command
        self._status = JarvisStatus.SPEAKING
        if self._app:
            self._app._toast("JARVIS: Listening...")

        def _ack_and_listen():
            try:
                if self._tts.available:
                    self._tts.speak_force(WAKE_ACK)
                    deadline = time.time() + 2.5
                    while not self._tts._queue.empty() and time.time() < deadline:
                        time.sleep(0.05)
                    time.sleep(0.2)  # Brief acoustic gap so mic doesn't capture TTS

                self._status = JarvisStatus.LISTENING
                if self._app:
                    self._app._toast("JARVIS: Listening for command...")
                self._stt.begin_listening()

                # Safety: wait for STT to finish, then ensure wake word resumes
                deadline = time.time() + 12.0  # max wait for STT to finish
                while self._stt.listening and time.time() < deadline:
                    time.sleep(0.1)
            except Exception as e:
                log.error(f"Error in wake ack+listen: {e}")
            finally:
                # Always resume wake word after STT finishes or times out
                if self._wwd and self._wwd.is_paused:
                    log.debug("Resuming wake-word detector after ack+listen.")
                    self._wwd.resume()
                if self._status == JarvisStatus.LISTENING:
                    self._status = JarvisStatus.READY

        threading.Thread(target=_ack_and_listen, daemon=True, name="jarvis-wake-ack").start()

    def _set_ready(self):
        if self._status == JarvisStatus.SPEAKING:
            self._status = JarvisStatus.LISTENING if (self._stt and self._stt.listening) else JarvisStatus.READY


# ── Command → Controller Dispatcher ──────────────────────────────────────────

def execute_command(cmd: VoiceCommand, controller, tts: "TextToSpeechService"):
    """
    Execute a VoiceCommand by calling the appropriate controller method,
    then speak the response.

    Called from the OpenCV main thread after drain_commands().

    Args:
        cmd:        The VoiceCommand to execute.
        controller: AeroScriptController instance.
        tts:        TextToSpeechService to speak the response.
    """
    intent = cmd.intent
    params = cmd.params
    tracker = cmd._tracker

    # Dispatch table — maps Intent → lambda(controller, params) → bool
    dispatch: dict[Intent, callable] = {
        Intent.CLEAR_CANVAS:       lambda c, p: c.clear_canvas(),
        Intent.UNDO:               lambda c, p: c.undo(),
        Intent.REDO:               lambda c, p: c.redo(),
        Intent.SAVE_CANVAS:        lambda c, p: c.save_canvas(),
        Intent.SCREENSHOT:         lambda c, p: c.screenshot(),
        Intent.CHANGE_COLOR:       lambda c, p: c.set_color(
                                        color_name=p.get("color_name", ""),
                                        color_idx=p.get("color_idx", -1)),
        Intent.INCREASE_BRUSH:     lambda c, p: c.increase_brush(),
        Intent.DECREASE_BRUSH:     lambda c, p: c.decrease_brush(),
        Intent.SELECT_PEN:         lambda c, p: c.select_tool("pen"),
        Intent.SELECT_ERASER:      lambda c, p: c.select_tool("eraser"),
        Intent.SELECT_HIGHLIGHTER: lambda c, p: c.select_tool("highlighter"),
        Intent.SELECT_MARKER:      lambda c, p: c.select_tool("marker"),
        Intent.ENABLE_MATH:        lambda c, p: c.select_tool("smart_math"),
        Intent.EXIT_MATH:          lambda c, p: c.select_tool("pen"),
        Intent.ENABLE_AIR_TEXT:    lambda c, p: c.select_tool("air_text"),
        Intent.EXIT_AIR_TEXT:      lambda c, p: c.select_tool("pen"),
        Intent.STAMP_RESULT:       lambda c, p: c.stamp_result(),
        Intent.PASTE_TO_SCREEN:    lambda c, p: c.paste_to_screen(),
        Intent.ENABLE_WHITEBOARD:  lambda c, p: c.enable_whiteboard(),
        Intent.ENABLE_CAMERA_MODE: lambda c, p: c.enable_camera_mode(),
        Intent.OPEN_GEMINI:        lambda c, p: c.open_gemini(),
        Intent.CLOSE_GEMINI:       lambda c, p: c.close_gemini(),
        Intent.SEARCH_GEMINI:      lambda c, p: c.search_gemini(p.get("query", "")),
        Intent.CLOSE_PROJECT:      lambda c, p: c.close_project(),
        Intent.START_PRESENTATION: lambda c, p: c.start_presentation(),
        Intent.STOP_PRESENTATION:  lambda c, p: c.stop_presentation(),
        Intent.START_ANNOTATION:   lambda c, p: c.start_annotation(),
        Intent.STOP_ANNOTATION:    lambda c, p: c.stop_annotation(),
        Intent.PAUSE:              lambda c, p: c.pause(),
        Intent.RESUME:             lambda c, p: c.resume(),
        Intent.SHOW_STATUS:        lambda c, p: c.show_status(),
        Intent.SHOW_FPS:           lambda c, p: c.show_fps(),
    }

    handler = dispatch.get(intent)
    if handler:
        try:
            if tracker:
                tracker.stamp("command_executed_start")
            handler(controller, params)
            if tracker:
                tracker.stamp("command_executed")
        except Exception as e:
            log.error(f"Controller error for {intent.name}: {e}")

        # Speak the response only when a handler actually ran
        color_name = params.get("color_name", "")
        query_text = params.get("query", "")
        response   = get_response(intent, color=color_name, query=query_text)
        if tracker:
            tracker.stamp("tts_queued")
        tts.speak(response)

        # Log full latency breakdown
        if tracker:
            tracker.log_summary()
    else:
        # UNKNOWN intent — tell the user what was heard so they can correct
        heard = cmd.raw_text[:48] if cmd.raw_text else "(nothing)"
        log.info(f"UNKNOWN intent — heard: '{heard}'")
        if controller and getattr(controller, "_app", None):
            controller._app._toast(f"JARVIS: '{heard}'?")
        if tracker:
            tracker.stamp("tts_queued")
        tts.speak(f"I heard: {heard}. Try again.")
        if tracker:
            tracker.log_summary()
