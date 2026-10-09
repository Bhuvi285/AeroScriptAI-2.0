"""
AeroScript 2.0 — State Manager
Centralized application state replaces the scattered global variables in test.py.
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from config import settings


@dataclass
class AppState:
    """
    Single source of truth for all mutable application state.

    Replaces the following globals from test.py:
        draw_color, brush_thickness, white_mode, eraser_mode,
        prev_x, prev_y, prev_pinch_x, prev_pinch_y, pinch_active, drawing_points
    And adds all new state for Phase 2+.
    """

    # ── Mode ──────────────────────────────────────────────────────────────────
    app_mode: str        = settings.MODE_CANVAS
    startup_done: bool   = False      # False = show startup screen

    # ── Tool & Color ─────────────────────────────────────────────────────────
    current_tool: str    = settings.TOOL_PEN
    current_color: Tuple = settings.DEFAULT_COLOR
    brush_size: int      = settings.DEFAULT_BRUSH_SIZE

    # ── Background ────────────────────────────────────────────────────────────
    white_mode: bool     = False      # True = white canvas background
    eraser_mode: bool    = False      # True = eraser active (tool shortcut)

    # ── Drawing state ─────────────────────────────────────────────────────────
    prev_x: Optional[int] = None
    prev_y: Optional[int] = None
    drawing_points: List  = field(default_factory=list)   # stroke point buffer for shape detect
    is_drawing: bool      = False

    # ── Pinch / Pan ───────────────────────────────────────────────────────────
    pinch_active: bool          = False
    prev_pinch_x: Optional[int] = None
    prev_pinch_y: Optional[int] = None

    # ── Gesture Tracking ──────────────────────────────────────────────────────
    current_gesture: str       = "none"
    gesture_confidence: float  = 0.0
    last_gesture_time: float   = field(default_factory=time.time)
    gesture_cooldown_ms: int   = settings.GESTURE_COOLDOWN_MS

    # Swipe detection state
    swipe_start_x: Optional[float] = None
    swipe_start_time: float        = 0.0

    # ── Handwriting Mode ──────────────────────────────────────────────────────
    handwriting_mode: bool     = False
    hw_stroke_points: List     = field(default_factory=list)
    hw_recognized_text: str    = ""
    hw_display_text: str       = ""    # text currently being built

    # ── Voice ─────────────────────────────────────────────────────────────────
    voice_enabled: bool        = settings.VOICE_ENABLED_DEFAULT
    last_voice_command: str    = ""
    voice_status_ts: float     = 0.0   # timestamp when command was received

    # ── Performance ───────────────────────────────────────────────────────────
    show_hud: bool             = settings.SHOW_FPS_DEFAULT
    show_debug: bool           = False

    # ── Accessibility ─────────────────────────────────────────────────────────
    accessibility_mode: bool   = False
    large_cursor: bool         = False
    high_contrast: bool        = False
    gesture_sensitivity: float = 1.0   # multiplier (1.0 = normal, >1 = more sensitive)

    # ── Presentation ─────────────────────────────────────────────────────────
    presentation_mode: bool    = False
    laser_pointer_active: bool = False

    # ── Annotation ────────────────────────────────────────────────────────────
    annotation_mode: bool      = False

    # ── Testing / Evaluation ─────────────────────────────────────────────────
    testing_mode: bool         = False
    expected_gesture: str      = ""

    # ── UI Transient ─────────────────────────────────────────────────────────
    status_message: str        = ""
    status_message_ts: float   = 0.0
    status_message_duration_ms: float = 2000.0

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def gesture_on_cooldown(self) -> bool:
        """Return True if the gesture cooldown has not elapsed yet."""
        elapsed_ms = (time.time() - self.last_gesture_time) * 1000
        return elapsed_ms < self.gesture_cooldown_ms

    def register_gesture(self, gesture: str, confidence: float = 1.0):
        """Update gesture state and cooldown timer."""
        self.current_gesture    = gesture
        self.gesture_confidence = confidence
        self.last_gesture_time  = time.time()

    def set_status(self, message: str, duration_ms: float = 2000.0):
        """Show a transient status message in the HUD."""
        self.status_message          = message
        self.status_message_ts       = time.time()
        self.status_message_duration_ms = duration_ms

    def get_active_status(self) -> Optional[str]:
        """Return status message if still within display window, else None."""
        elapsed_ms = (time.time() - self.status_message_ts) * 1000
        if elapsed_ms < self.status_message_duration_ms:
            return self.status_message
        return None

    def set_voice_command(self, command: str):
        """Record a recognized voice command for display."""
        self.last_voice_command = command
        self.voice_status_ts    = time.time()

    def get_voice_status(self) -> Optional[str]:
        """Return the last voice command if within display window."""
        elapsed_ms = (time.time() - self.voice_status_ts) * 1000
        if elapsed_ms < settings.VOICE_STATUS_DISPLAY_MS:
            return self.last_voice_command
        return None

    def reset_drawing(self):
        """Reset ephemeral drawing state between strokes."""
        self.prev_x        = None
        self.prev_y        = None
        self.drawing_points = []
        self.is_drawing    = False

    def effective_tool(self) -> str:
        """Return effective tool, accounting for eraser_mode override."""
        if self.eraser_mode:
            return settings.TOOL_ERASER
        return self.current_tool
