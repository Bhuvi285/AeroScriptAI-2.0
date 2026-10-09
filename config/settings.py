"""
AeroScript 2.0 — Central Configuration
All tunable constants are defined here. Import this module everywhere instead of hardcoding values.
"""

import os

# ─── Gemini AI Chatbot ────────────────────────────────────────────────────────
# Get a free key at: https://aistudio.google.com/app/apikey
# Set env var: export GEMINI_API_KEY="your-key-here"
def _resolve_gemini_key() -> str:
    k = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if k:
        return k
    for path in (".env", "config/gemini_key.txt"):
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            candidate = line.split("=", 1)[1].strip().strip('"').strip("'")
                            if candidate:
                                return candidate
                        elif not line.startswith("#") and len(line) >= 20:
                            return line.strip('"').strip("'")
            except Exception:
                pass
    return ""

GEMINI_API_KEY          = _resolve_gemini_key()
GEMINI_MODEL            = "gemini-flash-latest" # active free tier model
GEMINI_MAX_TOKENS       = 200                   # keep answers short for canvas stamping
GEMINI_CHAT_KEY         = ord('g')              # G key opens chat overlay

# ─── Camera ──────────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0          # Default webcam index
CAMERA_WIDTH        = 1280       # Capture width
CAMERA_HEIGHT       = 720        # Capture height
CAMERA_FPS          = 60         # Target capture FPS
CAMERA_FLIP         = True       # Horizontal mirror (selfie view)

# ─── MediaPipe Hand Tracking ─────────────────────────────────────────────────
MP_MAX_HANDS              = 1
MP_DETECTION_CONFIDENCE   = 0.7
MP_TRACKING_CONFIDENCE    = 0.7

# ─── Gesture Thresholds ──────────────────────────────────────────────────────
PINCH_DISTANCE_THRESHOLD  = 40   # px — index-thumb distance to activate pinch
GESTURE_COOLDOWN_MS       = 400  # ms — minimum time between gesture triggers
GESTURE_SMOOTHING_FRAMES  = 5    # frames — smoothing window for landmark positions
SWIPE_VELOCITY_THRESHOLD  = 0.04 # normalized — minimum hand velocity to count as swipe
FIST_CURL_THRESHOLD       = 0.6  # ratio — finger curl ratio for fist detection
PALM_SPREAD_THRESHOLD     = 0.08 # normalized — finger spread for open palm

# ─── Drawing ─────────────────────────────────────────────────────────────────
DEFAULT_BRUSH_SIZE    = 5        # px radius
ERASER_SIZE           = 30       # px radius
MAX_BRUSH_SIZE        = 40
MIN_BRUSH_SIZE        = 2
BRUSH_SIZE_STEP       = 2

# ─── Shape Detection ─────────────────────────────────────────────────────────
SHAPE_MIN_POINTS       = 25      # minimum stroke points to attempt shape recognition
SHAPE_APPROX_EPSILON   = 0.03    # Douglas-Peucker epsilon (fraction of perimeter)
CIRCLE_CIRCULARITY_MIN = 0.60    # minimum circularity ratio for circle detection
STAR_CONCAVITY_RATIO   = 0.40    # convexity defect ratio for star detection

# ─── Canvas ──────────────────────────────────────────────────────────────────
CANVAS_UNDO_LIMIT     = 50       # maximum undo history steps
AUTO_SAVE_INTERVAL_S  = 0        # disabled (only save when pressing S key)

# ─── Colors (BGR format for OpenCV) ──────────────────────────────────────────
COLOR_RED        = (0,   0,   200)
COLOR_GREEN      = (0,   200, 0)
COLOR_BLUE       = (200, 0,   0)
COLOR_WHITE      = (255, 255, 255)
COLOR_BLACK      = (0,   0,   0)
COLOR_YELLOW     = (0,   220, 255)
COLOR_ORANGE     = (0,   140, 255)
COLOR_PURPLE     = (180, 0,   180)
COLOR_CYAN       = (255, 200, 0)

# Predefined palette in order for toolbar slots
PALETTE = [
    ("Red",    COLOR_RED),
    ("Green",  COLOR_GREEN),
    ("Blue",   COLOR_BLUE),
    ("Yellow", COLOR_YELLOW),
    ("Orange", COLOR_ORANGE),
    ("Purple", COLOR_PURPLE),
    ("White",  COLOR_WHITE),
    ("Black",  COLOR_BLACK),
]

DEFAULT_COLOR = COLOR_RED

# ─── UI / Toolbar ────────────────────────────────────────────────────────────
TOOLBAR_HEIGHT      = 80         # px — top toolbar bar height
SIDE_PANEL_WIDTH    = 80         # px — right-side tool panel width
HUD_OPACITY         = 0.75       # alpha for overlay panels

# Apple-style UI palette (BGR)
UI_BG_DARK          = (30,  30,  30)    # #1E1E1E dark panel
UI_BG_PANEL         = (50,  50,  50)    # #323232 secondary panel
UI_ACCENT           = (200, 180, 0)     # #00B4C8 cyan accent (BGR)
UI_ACCENT_GREEN     = (50,  205, 100)   # green accent
UI_TEXT_PRIMARY     = (240, 240, 240)   # near-white
UI_TEXT_SECONDARY   = (160, 160, 160)   # grey
UI_SEPARATOR        = (70,  70,  70)    # divider lines
UI_SELECTED_BG      = (80,  80,  80)    # selected item highlight
UI_DANGER           = (40,  40,  220)   # red (BGR) for warnings/bin

# ─── Performance ─────────────────────────────────────────────────────────────
PERF_FPS_BUFFER_SIZE   = 30     # frames in FPS rolling average
SHOW_FPS_DEFAULT       = True
SHOW_GESTURE_DEFAULT   = True

# ─── Voice Commands ──────────────────────────────────────────────────────────
VOICE_ENABLED_DEFAULT     = False    # must be explicitly enabled
VOICE_ENERGY_THRESHOLD    = 300
VOICE_PAUSE_THRESHOLD     = 0.8     # seconds of silence to consider phrase complete (allows natural multi-syllable speech)
VOICE_STATUS_DISPLAY_MS   = 2500    # ms to show recognized command on screen

# ─── JARVIS Voice Assistant ───────────────────────────────────────────────────
JARVIS_ENABLED              = True           # enable JARVIS on startup
JARVIS_ALWAYS_LISTEN        = True           # always listen continuously in background
JARVIS_WAKE_WORD_ENABLED    = os.getenv("JARVIS_WAKE_WORD_ENABLED", "true").lower() in ("true", "1", "yes")
JARVIS_WAKE_WORDS           = [
    "hey jarvis", "ok jarvis", "okay jarvis", "jarvis",
    "wake up", "wake up jarvis", "wake up aeroscript",
    "hi jarvis", "hello jarvis", "aero script", "aeroscript",
]
JARVIS_LISTEN_TIMEOUT       = 5             # seconds to wait for speech before giving up (optimized: 7→5)
JARVIS_PHRASE_TIMEOUT       = 4             # seconds of silence to end a phrase (optimized: 5→4)
JARVIS_STT_ENERGY_THRESHOLD = 100           # PyAudio base energy threshold
JARVIS_STT_DYNAMIC_ENERGY   = True          # Adaptive ambient noise tracking
JARVIS_STT_LANGUAGE         = os.getenv("JARVIS_STT_LANGUAGE", "en-US") # Google STT recognition language tag
JARVIS_TTS_RATE             = 175           # words per minute
JARVIS_TTS_VOLUME           = 0.90          # TTS volume (0.0–1.0)
JARVIS_RESPONSE_COOLDOWN_S  = 3.0           # seconds before repeating same response
JARVIS_STARTUP_SPEECH       = True          # speak greeting on launch
JARVIS_PTT_KEY              = ord('v')      # push-to-talk key (V) (optional fallback)
JARVIS_STATUS_DISPLAY_MS    = 3000          # ms to show last JARVIS message in UI

# ─── Export ──────────────────────────────────────────────────────────────────
EXPORT_DIR         = "exports"
CANVAS_SAVE_DIR    = "data/canvases"
LOG_DIR            = "logs"
LOG_FILE           = "logs/aeroscript.log"

# ─── Application Modes ───────────────────────────────────────────────────────
MODE_CANVAS         = "canvas"
MODE_PRESENTATION   = "presentation"
MODE_ANNOTATION     = "annotation"
MODE_SETTINGS       = "settings"
MODE_TESTING        = "testing"
MODE_DEMO           = "demo"

# ─── Tools ───────────────────────────────────────────────────────────────────
TOOL_PEN            = "pen"
TOOL_PENCIL         = "pencil"
TOOL_MARKER         = "marker"
TOOL_HIGHLIGHTER    = "highlighter"
TOOL_ERASER         = "eraser"
TOOL_SHAPE_PEN      = "shape_pen"    # draws and auto-corrects shapes
TOOL_TEXT           = "text"

# ─── Handwriting Recognition ─────────────────────────────────────────────────
HWR_MODEL_PATH         = "models/emnist_cnn.pth"
HWR_MIN_STROKE_POINTS  = 20
HWR_INPUT_SIZE         = 28    # EMNIST image size
HWR_CONFIDENCE_MIN     = 0.55

# ─── Gesture Calibration Model ───────────────────────────────────────────────
GESTURE_MODEL_PATH         = "models/gesture_calibration.pkl"
GESTURE_TRAIN_SAMPLES      = 10    # samples per gesture for calibration

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL    = "INFO"    # DEBUG, INFO, WARNING, ERROR
