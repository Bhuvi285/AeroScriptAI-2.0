"""
AeroScript JARVIS — Centralised Response Strings
=================================================
All spoken and displayed JARVIS responses live here.
Keeps language out of logic files and makes localisation easy.
"""

from enum import Enum, auto


class Intent(Enum):
    """All supported voice intents."""
    # Canvas operations
    CLEAR_CANVAS        = auto()
    UNDO                = auto()
    REDO                = auto()
    SAVE_CANVAS         = auto()
    SCREENSHOT          = auto()

    # Color selection
    CHANGE_COLOR        = auto()

    # Brush
    INCREASE_BRUSH      = auto()
    DECREASE_BRUSH      = auto()

    # Tool selection
    SELECT_PEN          = auto()
    SELECT_ERASER       = auto()
    SELECT_HIGHLIGHTER  = auto()
    SELECT_MARKER       = auto()

    # Modes
    ENABLE_WHITEBOARD   = auto()
    ENABLE_CAMERA_MODE  = auto()
    ENABLE_MATH         = auto()
    EXIT_MATH           = auto()
    ENABLE_AIR_TEXT     = auto()
    EXIT_AIR_TEXT       = auto()
    STAMP_RESULT        = auto()
    PASTE_TO_SCREEN     = auto()
    OPEN_GEMINI         = auto()
    CLOSE_GEMINI        = auto()
    SEARCH_GEMINI       = auto()

    # Presentation / Annotation
    START_PRESENTATION  = auto()
    STOP_PRESENTATION   = auto()
    START_ANNOTATION    = auto()
    STOP_ANNOTATION     = auto()

    # Application control
    PAUSE               = auto()
    RESUME              = auto()
    CLOSE_PROJECT       = auto()

    # Info
    SHOW_STATUS         = auto()
    SHOW_FPS            = auto()

    # Special
    UNKNOWN             = auto()


# ── TTS Responses ─────────────────────────────────────────────────────────────
# Maps Intent → spoken response string.
# {color} is replaced by the detected color name at runtime.

RESPONSES: dict[Intent, str] = {
    Intent.CLEAR_CANVAS:       "Canvas cleared.",
    Intent.UNDO:               "Undo is disabled.",
    Intent.REDO:               "Redo is disabled.",
    Intent.SAVE_CANVAS:        "Canvas saved successfully.",
    Intent.SCREENSHOT:         "Screenshot saved.",
    Intent.CHANGE_COLOR:       "{color} selected.",
    Intent.INCREASE_BRUSH:     "Brush size increased.",
    Intent.DECREASE_BRUSH:     "Brush size decreased.",
    Intent.SELECT_PEN:         "Pen selected.",
    Intent.SELECT_ERASER:      "Eraser selected.",
    Intent.SELECT_HIGHLIGHTER: "Highlighter selected.",
    Intent.SELECT_MARKER:      "Marker selected.",
    Intent.ENABLE_WHITEBOARD:  "Whiteboard mode enabled.",
    Intent.ENABLE_CAMERA_MODE: "Camera mode enabled.",
    Intent.ENABLE_MATH:        "Smart Math mode enabled.",
    Intent.EXIT_MATH:          "Smart Math exited. Pen selected.",
    Intent.ENABLE_AIR_TEXT:    "Air writing mode enabled.",
    Intent.EXIT_AIR_TEXT:      "Air writing exited. Pen selected.",
    Intent.STAMP_RESULT:       "Result stamped on canvas.",
    Intent.PASTE_TO_SCREEN:    "Pasting answer onto screen.",
    Intent.OPEN_GEMINI:        "Gemini AI chat opened.",
    Intent.CLOSE_GEMINI:       "Gemini AI chat closed.",
    Intent.SEARCH_GEMINI:      "Searching Gemini for {query}.",
    Intent.START_PRESENTATION: "Presentation mode activated.",
    Intent.STOP_PRESENTATION:  "Presentation mode stopped.",
    Intent.START_ANNOTATION:   "Annotation started.",
    Intent.STOP_ANNOTATION:    "Annotation stopped.",
    Intent.PAUSE:              "Paused.",
    Intent.RESUME:             "Resuming.",
    Intent.CLOSE_PROJECT:      "Closing AeroScript. Goodbye.",
    Intent.SHOW_STATUS:        "Status displayed.",
    Intent.SHOW_FPS:           "FPS displayed.",
    Intent.UNKNOWN:            "Sorry, I didn't understand that command.",
}

# ── Toast Labels ──────────────────────────────────────────────────────────────
# Short labels shown in the OpenCV status toast (on-screen text).

TOAST_LABELS: dict[Intent, str] = {
    Intent.CLEAR_CANVAS:       "JARVIS: Canvas cleared",
    Intent.UNDO:               "JARVIS: Undo disabled",
    Intent.REDO:               "JARVIS: Redo disabled",
    Intent.SAVE_CANVAS:        "JARVIS: Canvas saved",
    Intent.SCREENSHOT:         "JARVIS: Screenshot saved",
    Intent.CHANGE_COLOR:       "JARVIS: Color → {color}",
    Intent.INCREASE_BRUSH:     "JARVIS: Brush ▲",
    Intent.DECREASE_BRUSH:     "JARVIS: Brush ▼",
    Intent.SELECT_PEN:         "JARVIS: Pen",
    Intent.SELECT_ERASER:      "JARVIS: Eraser",
    Intent.SELECT_HIGHLIGHTER: "JARVIS: Highlighter",
    Intent.SELECT_MARKER:      "JARVIS: Marker",
    Intent.ENABLE_WHITEBOARD:  "JARVIS: Whiteboard ON",
    Intent.ENABLE_CAMERA_MODE: "JARVIS: Camera mode",
    Intent.ENABLE_MATH:        "JARVIS: Smart Math ON",
    Intent.EXIT_MATH:          "JARVIS: Pen Mode",
    Intent.ENABLE_AIR_TEXT:    "JARVIS: Air-Text ON",
    Intent.EXIT_AIR_TEXT:      "JARVIS: Pen Mode",
    Intent.STAMP_RESULT:       "JARVIS: Stamped",
    Intent.PASTE_TO_SCREEN:    "JARVIS: Pasted to Screen",
    Intent.OPEN_GEMINI:        "JARVIS: Gemini Chat ON",
    Intent.CLOSE_GEMINI:       "JARVIS: Gemini Chat OFF",
    Intent.SEARCH_GEMINI:      "JARVIS: Searching Gemini...",
    Intent.START_PRESENTATION: "JARVIS: Presentation mode",
    Intent.STOP_PRESENTATION:  "JARVIS: Presentation stopped",
    Intent.START_ANNOTATION:   "JARVIS: Annotation ON",
    Intent.STOP_ANNOTATION:    "JARVIS: Annotation OFF",
    Intent.PAUSE:              "JARVIS: Paused",
    Intent.RESUME:             "JARVIS: Resumed",
    Intent.CLOSE_PROJECT:      "JARVIS: Closing AeroScript...",
    Intent.SHOW_STATUS:        "JARVIS: Status",
    Intent.SHOW_FPS:           "JARVIS: FPS",
    Intent.UNKNOWN:            "JARVIS: ?",
}

# ── System Messages ───────────────────────────────────────────────────────────
STARTUP_GREETING         = "Good morning. AeroScript is ready."
STARTUP_GREETING_SIMPLE  = "AeroScript ready."
WAKE_ACK                 = "Yes?"
LISTENING_START          = "JARVIS listening."
NO_MICROPHONE            = "Microphone unavailable. Voice control disabled."
NETWORK_ERROR            = "Speech recognition unavailable. Check your connection."
TTS_ERROR                = "Text to speech unavailable."
RECOGNITION_TIMEOUT      = "I didn't catch that. Please try again."


def get_response(intent: "Intent", **kwargs) -> str:
    """Return the TTS response string for an intent, with optional substitutions."""
    template = RESPONSES.get(intent, RESPONSES[Intent.UNKNOWN])
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def get_toast(intent: "Intent", **kwargs) -> str:
    """Return the toast label string for an intent, with optional substitutions."""
    template = TOAST_LABELS.get(intent, TOAST_LABELS[Intent.UNKNOWN])
    try:
        return template.format(**kwargs)
    except KeyError:
        return template
