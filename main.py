"""
AeroScript 2.0 — AI-Powered Touchless Human-Computer Interaction System
========================================================================
ALL UI interactions are keyboard-driven. No mouse required.
Drawing happens purely via hand gestures (MediaPipe).

KEYBOARD CONTROLS:
──────────────────
  1-7       → Select color (Red, Green, Blue, Yellow, Orange, White, Black)
  T         → Cycle tool  (Pen → Marker → Highlighter → Shape Pen → Air-Text → Smart Math)
  X         → Direct toggle Air-Writing / Text Conversion mode (TXT)
  M         → Direct toggle Smart Mathematical Drawing mode (MTH)
  Enter     → Solve / Stamp recognized text or math equation onto canvas
  E         → Toggle Eraser
  W         → Toggle Whiteboard (white bg) / Camera mode
  C         → Clear canvas
  V         → Push-to-Talk JARVIS Assistant
  G         → Toggle Gemini AI Chatbot
  +  or =   → Brush size up
  -         → Brush size down
  S         → Save PNG to exports/
  D         → Toggle debug panel (FPS / Latency / CPU)
  H         → Toggle HUD
  ESC       → Exit

GESTURE CONTROLS (hand in front of camera):
────────────────────────────────────────────
  Index finger up           → Draw
  Index + Middle fingers up → Selection mode (no drawing)
  Pinch (thumb + index)     → Pan canvas
  Pinch + drag to BIN zone  → Clear entire canvas

AI FEATURES:
────────────
  1. Air-Writing → Text Conversion (Key X or T to TXT):
     Write letters in air (e.g. HELLO, AERO) → AI CNN classifies & displays text
  2. Smart Mathematical Drawing (Key M or T to MTH):
     Draw arithmetic expressions (e.g. 2 + 3, 12 * 4) → AI recognizes & solves (e.g. 5, 48)
  3. Shape Auto-Correction (T to SHP):
     Draw rough Circle, Square, Rectangle, Triangle → auto-corrects on finger lower
"""

import cv2
import numpy as np
import logging
import os
import sys
import time
import ui_icons
from collections import deque

# ── JARVIS Voice Assistant ────────────────────────────────────────────────────
try:
    from assistant.jarvis     import JarvisAssistant, execute_command, JarvisStatus, STATUS_COLORS, STATUS_LABELS
    from assistant.controller import AeroScriptController
    from config               import settings as _cfg
    _JARVIS_AVAILABLE = True
except ImportError as _e:
    _JARVIS_AVAILABLE = False
    import logging as _l; _l.getLogger("AeroScript").warning(f"JARVIS not available: {_e}")

# ── Paths & Logging ───────────────────────────────────────────────────────────
os.makedirs("logs",            exist_ok=True)
os.makedirs("exports",         exist_ok=True)
os.makedirs("data/canvases",   exist_ok=True)
os.makedirs("models",          exist_ok=True)

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/aeroscript.log", encoding="utf-8"),
        logging.StreamHandler(
            open(sys.stdout.fileno(), mode="w", encoding="utf-8", errors="replace", closefd=False)
        ),
    ]
)
log = logging.getLogger("AeroScript")

# ── Core & Vision & AI ────────────────────────────────────────────────────────
import mediapipe as mp
from core.canvas           import Canvas
from core.drawing_engine   import DrawingEngine
from vision.hand_tracker   import HandTracker
from vision.shape_detector import detect_shape, draw_perfect_shape, SHAPE_UNKNOWN
from ai.air_writing        import AirWritingEngine
from ai.math_engine        import MathEngine
from ai.gemini_chat        import GeminiChatEngine

# ── Constants ─────────────────────────────────────────────────────────────────
WIN   = "AeroScript 2.0"
W, H  = 1280, 720

FONT  = cv2.FONT_HERSHEY_SIMPLEX
FONTB = cv2.FONT_HERSHEY_DUPLEX

# Colors (BGR) — mapped to keys 1-7
COLORS = [
    ("Red",    (0,   0,   220)),
    ("Green",  (0,   200,  50)),
    ("Blue",   (200,  30,   0)),
    ("Yellow", (0,   220, 255)),
    ("Orange", (0,   140, 255)),
    ("White",  (240, 240, 240)),
    ("Black",  (20,   20,  20)),
]

# Tools cycled with T key
TOOLS = ["pen", "marker", "highlighter", "shape_pen", "air_text", "smart_math"]
TOOL_LABELS = {
    "pen":        "Pen",
    "marker":     "Marker",
    "highlighter":"Highlighter",
    "shape_pen":  "Shape Pen",
    "air_text":   "Air-Text",
    "smart_math": "Smart Math",
}

# UI palette
_DARK       = (20,  20,  24)   # #181414 deep dark obsidian
_DARK_GLASS = (14,  14,  18)   # frosted glass base
_PANEL      = (34,  36,  44)   # secondary card/panel surface
_PANEL_HOV  = (48,  52,  64)   # hover pill
_CYAN       = (235, 180,   0)  # #00B4EB electric cyan (BGR)
_CYAN_DIM   = (100,  80,  20)  # dim cyan glow
_GREEN      = ( 60, 215, 110)  # #6ED73C vibrant emerald
_RED        = ( 55,  55, 235)  # #EB3737 electric coral
_ORANGE     = (  0, 150, 255)  # vivid amber
_PURPLE     = (220,  80, 160)  # modern violet
_WHITE      = (245, 245, 248)  # crisp near-white
_GREY       = (160, 165, 175)  # subtle label grey
_DIM        = ( 95, 100, 112)  # divider / hint grey
_SEP        = ( 50,  54,  66)  # separator line
_BORDER     = ( 50,  54,  66)  # sleek 1px container border
_BORDER_LGT = ( 75,  80,  98)  # container highlight

BAR_H  = 64   # top bar height
BOT_H  = 34   # bottom hint bar


def _rrect(img, x1, y1, x2, y2, col, r=8, border_col=None, border_thick=1):
    """Smooth anti-aliased filled rounded rectangle with optional border."""
    r = max(2, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    AA = cv2.LINE_AA
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), col, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), col, -1)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, col, -1, AA)
    if border_col is not None:
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, border_col, border_thick, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, border_col, border_thick, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, border_col, border_thick, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, border_col, border_thick, AA)
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), border_col, border_thick, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), border_col, border_thick, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), border_col, border_thick, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), border_col, border_thick, AA)


def _glass_panel(img, x1, y1, x2, y2, col=_DARK_GLASS, alpha=0.88, border_col=_BORDER, r=8):
    """Draws a smooth frosted glass panel overlay with alpha blending and optional rounded border."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return
    roi = img[y1:y2, x1:x2]
    patch = np.full((h, w, 3), col, dtype=np.uint8)
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi
    if border_col is not None:
        _rrect(img, x1, y1, x2, y2, col, r=r, border_col=border_col)







class App:
    def __init__(self):
        # State
        self.color_idx   = 0
        self.tool_idx    = 0
        self.brush       = 5
        self.eraser      = False
        self.white_mode  = False
        self.show_hud    = True
        self.show_debug  = False

        # Gesture state
        self.prev_x      = None
        self.prev_y      = None
        self.pts         = []        # stroke buffer for shape detect
        self.is_drawing  = False
        self.pinch       = False
        self.ppx = self.ppy = None   # previous pinch point

        # Canvas
        self.canvas      = None
        self.white_bg    = None

        # AI Engine & State
        self.ai_engine       = AirWritingEngine()
        self.ai_result       = None
        self.last_draw_time  = 0.0
        self.ai_solved       = False

        # Status toast
        self.toast       = ""
        self.toast_t     = 0.0

        # Perf
        self.frame_ts    = deque(maxlen=120)  # recent frame times for FPS
        self.lat_ms      = 0.0
        self._frame_idx  = 0           # frame counter for skip logic

        # JARVIS Voice Assistant & Controller
        self.jarvis:      "JarvisAssistant | None" = None
        self.controller   = AeroScriptController(self) if _JARVIS_AVAILABLE else None
        self.jarvis_paused = False   # set by voice 'pause' command
        self.mic_bbox      = None    # (x1, y1, x2, y2) for UI hit testing
        
        # Side Menu & UI Hit Testing
        self.mouse_pos        = (-1, -1)
        self.menu_open        = False
        self.menu_btn_bbox    = None
        self.menu_panel_bbox  = None
        self.menu_items       = []      # stores (bbox, lambda) for hit testing
        self.swatch_bboxes    = []      # stores (bbox, color_idx)
        self.tool_pill_bbox   = None
        self.wb_pill_bbox     = None
        self.brush_plus_bbox  = None
        self.brush_minus_bbox = None
        self.is_selection_mode = False
        self._last_finger_select_time = 0.0

        # ── Gemini AI Chatbot ───────────────────────────────────────────────────
        self.gemini       = GeminiChatEngine(api_key=_cfg.GEMINI_API_KEY if _JARVIS_AVAILABLE else "")
        self.chat_mode    = False    # True = chat overlay is open
        self.chat_input   = ""      # current typed question
        self.chat_answer  = ""      # latest Gemini answer
        self.chat_thinking= False   # True while waiting for Gemini
        self.chat_stamped = False   # True after answer stamped to canvas
        self.chat_api_input = False # True = user is typing API key
        self._last_chat_query = ""
        self.running      = True    # app lifecycle flag controlled by voice or ESC

        # Hand Tracker (replaces inline MediaPipe usage)
        self.hand_tracker = HandTracker()

    # ── Main entry ────────────────────────────────────────────────────────────

    def run(self):
        # Show startup
        mode = self._startup()
        if mode == "exit":
            return

        # ── JARVIS Initialisation ──────────────────────────────────────────
        self._init_jarvis()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._err("Camera not found.\nConnect a webcam and press any key.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_FPS, 60)

        # Try to use DirectShow/MSMF backend for lowest latency on Windows
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Initialize HandTracker (uses config settings for confidence thresholds)
        self.hand_tracker.start()
        mp_drawing = mp.solutions.drawing_utils
        mp_styles  = mp.solutions.drawing_styles
        mp_hands   = mp.solutions.hands

        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, W, H)
            cv2.setMouseCallback(WIN, self._mouse_callback)
            log.info("Canvas loop started.")

            while self.running:
                t0 = time.perf_counter()
                now = time.time()

                # ── JARVIS: drain command queue (non-blocking) ─────────────
                if self.jarvis and self.controller:
                    for cmd in self.jarvis.drain_commands():
                        execute_command(cmd, self.controller, self.jarvis._tts)

                if not self.running:
                    break

                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)
                fh, fw = frame.shape[:2]
                self._frame_idx += 1

                # Lazy init canvas
                if self.canvas is None:
                    self.canvas   = Canvas(fh, fw)
                    self.white_bg = np.ones((fh, fw, 3), dtype=np.uint8) * 255
                    log.info(f"Canvas init {fw}x{fh}")

                # ── Hand tracking (skip every other frame for speed) ──────
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self._frame_idx % 2 == 0:
                    hand_data = self.hand_tracker.process(frame)
                    self._last_hand_data = hand_data
                else:
                    hand_data = getattr(self, '_last_hand_data', None)

                hand_present = False
                cx = cy = tx = ty = 0
                index_up = middle_up = False

                if hand_data:
                    hand_present = True
                    lm_px = hand_data['landmarks_px']
                    lm_norm = hand_data['landmarks_norm']

                    (cx, cy) = lm_px[8]     # index tip
                    (tx, ty) = lm_px[4]     # thumb tip
                    _, iy6   = lm_px[6]     # index PIP y
                    _, iy12  = lm_px[10]    # middle PIP y
                    my       = lm_px[12][1] # middle tip y

                    index_up  = cy < iy6
                    middle_up = my < iy12


                # ── Gesture logic ─────────────────────────────────────────
                if hand_present:
                    dist = np.hypot(cx-tx, cy-ty)
                    is_pinch = dist < 40

                    if self.menu_open:
                        # Allow finger interaction / selection inside open menu
                        if is_pinch or (index_up and middle_up) or index_up:
                            self._handle_finger_interaction(cx, cy, now)
                    elif is_pinch:
                        # Pan
                        self.pinch = True
                        if self.ppx is not None:
                            self.canvas.pan(cx-self.ppx, cy-self.ppy)
                        self.ppx, self.ppy = cx, cy
                        self._reset_draw()

                        # BIN zone (bottom centre)
                        bx1 = fw//2-55; by1 = fh-56
                        bx2 = fw//2+55; by2 = fh
                        if bx1 < cx < bx2 and by1 < cy < by2:
                            self.canvas.clear()
                            self.ai_result = None
                            self.ai_solved = False
                            self._toast("Canvas cleared!")

                    else:
                        self.pinch = False
                        self.ppx = self.ppy = None

                        if index_up and middle_up:
                            # ── Two-finger Selection Mode (Peace Sign ✌️) ──
                            self.is_selection_mode = True
                            self._reset_draw()
                            self._handle_finger_interaction(cx, cy, now)

                        elif index_up:
                            # ── One-finger Interaction ──
                            if cy <= BAR_H + 20:
                                # Finger in top dock: direct air touch/selection without drawing stray ink!
                                self.is_selection_mode = True
                                self._reset_draw()
                                self._handle_finger_interaction(cx, cy, now)
                            else:
                                self.is_selection_mode = False
                                # DRAW on canvas
                                self.pts.append((cx, cy))
                                self.is_drawing = True
                                self.last_draw_time = time.time()
                                self.ai_solved = False
                                tool = self._tool()

                                if self.prev_x is not None:
                                    if len(self.pts) == 2:
                                        self.canvas.save_snapshot()
                                    if self.eraser:
                                        DrawingEngine.erase_stroke(
                                            self.canvas.surface,
                                            self.prev_x, self.prev_y, cx, cy)
                                    else:
                                        # If in air_text or smart_math, draw clean crisp stroke
                                        active_tool = "pen" if tool in ("air_text", "smart_math") else tool
                                        DrawingEngine.draw_stroke(
                                            self.canvas.surface,
                                            self.prev_x, self.prev_y, cx, cy,
                                            self._color(), self.brush, active_tool)
                                    self.canvas._mark_dirty()
                                self.prev_x, self.prev_y = cx, cy

                        else:
                            # Finger lowered
                            self.is_selection_mode = False
                            if self.is_drawing and self._tool() == "shape_pen":
                                self._shape_detect()
                            self._reset_draw()
                else:
                    self.is_selection_mode = False
                    self._reset_draw()
                    self.pinch = False
                    self.ppx = self.ppy = None

                # ── AI Auto-Recognition (1.0s pause after drawing) ────────
                if (self._tool() in ("air_text", "smart_math") and
                        not self.is_drawing and not self.ai_solved and
                        self.last_draw_time > 0 and (now - self.last_draw_time > 1.0)):
                    self._run_ai_solve()
                    self.ai_solved = True

                # ── Composite ─────────────────────────────────────────────
                bg     = self.white_bg if self.white_mode else frame
                output = self.canvas.composite(bg)

                # ── UI rendering ──────────────────────────────────────────
                self._draw_ui(output, hand_present, cx, cy, fw, fh)

                # ── Perf ──────────────────────────────────────────────────
                self.lat_ms = (time.perf_counter() - t0) * 1000
                now2 = time.perf_counter()
                self.frame_ts.append(now2)
                # Trim old timestamps (deque handles maxlen, just count recent)
                self._fps = sum(1 for t in self.frame_ts if now2 - t < 1.0)

                cv2.imshow(WIN, output)

                # ── Keyboard ──────────────────────────────────────────────
                k = cv2.waitKey(1) & 0xFF
                if not self._handle_key(k):
                    break
                if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                    break

        finally:
            self.hand_tracker.stop()
        cap.release()
        cv2.destroyAllWindows()
        if self.jarvis:
            self.jarvis.stop()
        log.info("Exit.")

    # ── AI Recognition & Stamping ─────────────────────────────────────────────

    def _run_ai_solve(self):
        """Runs character segmentation, CNN classification, and math evaluation."""
        if not self.canvas:
            return
        mode = "math" if self._tool() == "smart_math" else "text"
        res  = self.ai_engine.process_canvas(self.canvas.surface, mode=mode)
        self.ai_result = res
        if res.get("text"):
            self._toast(f"AI: {res['text']}")
            log.info(f"AI [{mode.upper()}]: {res['text']} (conf {res.get('confidence',0)*100:.1f}%)")

    def _stamp_ai_result(self):
        """Stamps recognized text or mathematical answer directly onto canvas."""
        if not self.canvas or not self.ai_result or not self.ai_result.get("text"):
            self._run_ai_solve()
            return

        tool = self._tool()
        boxes = self.ai_result.get("boxes", [])

        if tool == "smart_math":
            math_sol = self.ai_result.get("math_solution")
            if math_sol and math_sol["valid"]:
                stamp_text = f" = {math_sol['result']}"
                if boxes:
                    # Find rightmost box
                    rightmost = max(boxes, key=lambda b: b[0] + b[2])
                    rx = min(rightmost[0] + rightmost[2] + 16, W - 220)
                    ry = rightmost[1]
                else:
                    rx, ry = 100, 200

                self.canvas.save_snapshot()
                draw_color = self._color()
                self._draw_handwritten_text(
                    self.canvas.surface, stamp_text, rx, ry, draw_color, font_size=58)
                self._toast(f"Stamped: {stamp_text}")
                self.ai_solved = True

        elif tool == "air_text":
            text = self.ai_result.get("text", "")
            if text:
                if boxes:
                    min_x = min(b[0] for b in boxes)
                    max_y = max(b[1] + b[3] for b in boxes)
                    rx = max(10, min_x)
                    ry = min(H - 40, max_y + 42)
                else:
                    rx, ry = 100, 200

                self.canvas.save_snapshot()
                # Render with handwritten font in current draw color
                draw_color = self._color()   # BGR
                self._draw_handwritten_text(
                    self.canvas.surface, text, rx, ry, draw_color, font_size=64)
                self._toast(f"Stamped: {text}")
                self.ai_solved = True

    # ── UI Rendering ──────────────────────────────────────────────────────────

    def _draw_ui(self, out, hand_present, cx, cy, fw, fh):
        mx, my = getattr(self, 'mouse_pos', (-1, -1))
        # Effective pointer coordinates for hover highlights (active finger or mouse)
        hx, hy = (cx, cy) if (hand_present and (self.is_selection_mode or cy <= BAR_H + 20)) else (-1, -1)
        eff_mx, eff_my = (hx, hy) if hx >= 0 else (mx, my)
        curr_tool = self._tool()

        # ── Top Frosted Glass Dock ─────────────────────────────────────────
        _glass_panel(out, 0, 0, fw, BAR_H, col=_DARK_GLASS, alpha=0.92, border_col=_BORDER, r=0)
        cv2.line(out, (0, 0), (fw, 0), _BORDER_LGT, 1, cv2.LINE_AA)
        cv2.line(out, (0, BAR_H), (fw, BAR_H), _BORDER, 1, cv2.LINE_AA)

        # ── 1. Hamburger Menu Button ───────────────────────────────────────
        hb_x, hb_y, hb_size = 14, 14, 36
        self.menu_btn_bbox = (hb_x, hb_y, hb_x + hb_size, hb_y + hb_size)
        hb_hover = (hb_x <= eff_mx <= hb_x + hb_size and hb_y <= eff_my <= hb_y + hb_size)
        hb_bg = _PANEL_HOV if (hb_hover or self.menu_open) else _PANEL
        hb_border = _CYAN if (hb_hover or self.menu_open) else _BORDER
        _rrect(out, hb_x, hb_y, hb_x + hb_size, hb_y + hb_size, hb_bg, r=8, border_col=hb_border)
        ui_icons.draw_icon(out, "hamburger", hb_x + hb_size // 2, hb_y + hb_size // 2, 18,
                           _CYAN if (hb_hover or self.menu_open) else _WHITE, 2)

        # ── 2. Brand + Live FPS Badge ─────────────────────────────────────
        brand_x = hb_x + 48
        cv2.putText(out, "AeroScript", (brand_x, 34), FONTB, 0.62, _WHITE, 2, cv2.LINE_AA)
        cv2.circle(out, (brand_x + 124, 26), 3, _CYAN, -1, cv2.LINE_AA)

        fps = getattr(self, '_fps', 0)
        fps_col = _GREEN if fps >= 30 else _ORANGE
        fps_bg  = (22, 38, 28) if fps >= 30 else (38, 30, 22)
        _rrect(out, brand_x, 42, brand_x + 58, 58, fps_bg, r=4, border_col=fps_col)
        cv2.putText(out, f"{fps} FPS", (brand_x + 6, 54), FONTB, 0.35, fps_col, 1, cv2.LINE_AA)

        # ── 3. Interactive Color Palette Dock ─────────────────────────────
        dock_x = brand_x + 150
        SW = 20; SM = 7
        dock_w = len(COLORS) * (SW + SM) + 12
        _rrect(out, dock_x, 14, dock_x + dock_w, BAR_H - 14, (26, 28, 34), r=18, border_col=_BORDER)

        self.swatch_bboxes.clear()
        for i, (name, col) in enumerate(COLORS):
            ccx = dock_x + 10 + i * (SW + SM) + SW // 2
            ccy = BAR_H // 2
            bx1, bx2 = ccx - (SW + SM) // 2, ccx + (SW + SM) // 2
            by1, by2 = 8, BAR_H - 8
            is_swatch_hover = (bx1 <= eff_mx <= bx2 and by1 <= eff_my <= by2)
            is_selected     = (i == self.color_idx and not self.eraser)

            # Hitbox registration (generous hit target for effortless mid-air finger interaction)
            self.swatch_bboxes.append(((bx1, by1, bx2, by2), i))

            if is_selected:
                cv2.circle(out, (ccx, ccy), SW // 2 + 5, _CYAN, 2, cv2.LINE_AA)
                cv2.circle(out, (ccx, ccy), SW // 2, col, -1, cv2.LINE_AA)
                cv2.circle(out, (ccx, ccy), 2, _WHITE, -1, cv2.LINE_AA)
            else:
                if is_swatch_hover:
                    cv2.circle(out, (ccx, ccy), SW // 2 + 4, _WHITE, 2, cv2.LINE_AA)
                cv2.circle(out, (ccx, ccy), SW // 2, col, -1, cv2.LINE_AA)
                cv2.circle(out, (ccx, ccy), SW // 2, _BORDER, 1, cv2.LINE_AA)

        # ── 4. Interactive Active Tool Capsule ─────────────────────────────
        icon_name = "eraser" if self.eraser else curr_tool
        tlabel = "Eraser" if self.eraser else TOOL_LABELS.get(curr_tool, "Pen")

        if self.eraser:
            tcol = _RED; pill_bg = (48, 24, 28); pill_border = (180, 50, 60)
        elif curr_tool == "smart_math":
            tcol = _GREEN; pill_bg = (24, 46, 32); pill_border = (50, 180, 90)
        elif curr_tool == "air_text":
            tcol = _ORANGE; pill_bg = (46, 38, 22); pill_border = (200, 150, 30)
        else:
            tcol = _CYAN; pill_bg = (22, 40, 50); pill_border = (0, 160, 200)

        tw = cv2.getTextSize(tlabel, FONTB, 0.48, 1)[0][0]
        pill_x = dock_x + dock_w + 14
        pill_h = 36
        pill_y = (BAR_H - pill_h) // 2
        pill_w = tw + 46
        self.tool_pill_bbox = (pill_x, pill_y, pill_x + pill_w, pill_y + pill_h)

        is_tool_hover = (pill_x <= eff_mx <= pill_x + pill_w and pill_y <= eff_my <= pill_y + pill_h)
        if is_tool_hover:
            pill_border = _WHITE

        _rrect(out, pill_x, pill_y, pill_x + pill_w, pill_y + pill_h, pill_bg, r=10, border_col=pill_border)
        # Rounded icon well
        _rrect(out, pill_x + 4, pill_y + 4, pill_x + 30, pill_y + 32, (18, 20, 26), r=6)
        ui_icons.draw_icon(out, icon_name, pill_x + 17, BAR_H // 2, 14, tcol, 2)
        cv2.putText(out, tlabel, (pill_x + 36, BAR_H // 2 + 5), FONTB, 0.48, _WHITE, 1, cv2.LINE_AA)

        # ── 5. Interactive Whiteboard Toggle Pill ─────────────────────────
        wbx = pill_x + pill_w + 12
        wb_w = 40; wb_h = 36; wb_y = (BAR_H - wb_h) // 2
        self.wb_pill_bbox = (wbx, wb_y, wbx + wb_w, wb_y + wb_h)
        is_wb_hover = (wbx <= eff_mx <= wbx + wb_w and wb_y <= eff_my <= wb_y + wb_h)

        if self.white_mode:
            wbc_bg = (235, 235, 240); wbc_ic = (30, 32, 38); wbc_border = _WHITE
        else:
            wbc_bg = _PANEL_HOV if is_wb_hover else _PANEL
            wbc_ic = _CYAN if is_wb_hover else _GREY
            wbc_border = _CYAN if is_wb_hover else _BORDER

        _rrect(out, wbx, wb_y, wbx + wb_w, wb_y + wb_h, wbc_bg, r=8, border_col=wbc_border)
        ui_icons.draw_icon(out, "whiteboard", wbx + wb_w // 2, BAR_H // 2, 16, wbc_ic, 2)

        # ── 6. Interactive Brush Size Pill with Live Preview Dot ──────────
        bsx = wbx + wb_w + 12
        bsw = 76; bsh = 36; bsy = (BAR_H - bsh) // 2
        _rrect(out, bsx, bsy, bsx + bsw, bsy + bsh, _PANEL, r=8, border_col=_BORDER)

        # Minus button
        self.brush_minus_bbox = (bsx + 2, bsy + 2, bsx + 22, bsy + bsh - 2)
        is_minus_hover = (bsx + 2 <= eff_mx <= bsx + 22 and bsy + 2 <= eff_my <= bsy + bsh - 2)
        cv2.putText(out, "-", (bsx + 8, BAR_H // 2 + 5), FONTB, 0.52, _CYAN if is_minus_hover else _GREY, 1, cv2.LINE_AA)

        # Live thickness preview circle (matching current brush thickness & draw color!)
        preview_r = max(2, min(self.brush // 2, 7))
        cv2.circle(out, (bsx + bsw // 2, BAR_H // 2), preview_r, self._color() if not self.eraser else _WHITE, -1, cv2.LINE_AA)

        # Plus button
        self.brush_plus_bbox = (bsx + bsw - 22, bsy + 2, bsx + bsw - 2, bsy + bsh - 2)
        is_plus_hover = (bsx + bsw - 22 <= eff_mx <= bsx + bsw - 2 and bsy + 2 <= eff_my <= bsy + bsh - 2)
        cv2.putText(out, "+", (bsx + bsw - 16, BAR_H // 2 + 5), FONTB, 0.50, _CYAN if is_plus_hover else _GREY, 1, cv2.LINE_AA)

        # ── 7. Hand Status Capsule (Far Right) ────────────────────────────
        hand_w = 98; hand_h = 32; hand_y = (BAR_H - hand_h) // 2
        hand_x = fw - hand_w - 14
        h_bg     = (24, 38, 28) if hand_present else (30, 30, 36)
        h_border = (45, 120, 65) if hand_present else _BORDER
        h_text   = "HAND ON" if hand_present else "NO HAND"
        h_col    = _GREEN if hand_present else _DIM

        _rrect(out, hand_x, hand_y, hand_x + hand_w, hand_y + hand_h, h_bg, r=16, border_col=h_border)
        cv2.circle(out, (hand_x + 14, BAR_H // 2), 4, h_col, -1, cv2.LINE_AA)
        cv2.putText(out, h_text, (hand_x + 26, BAR_H // 2 + 4), FONTB, 0.38, h_col, 1, cv2.LINE_AA)

        # ── 8. JARVIS Voice Assistant Pill ────────────────────────────────
        if self.jarvis and _JARVIS_AVAILABLE:
            self._draw_mic_button(out, fw, fh, hand_x, eff_mx, eff_my)

            # Floating Last Recognized Speech Pill
            if (self.jarvis.last_text and time.time() - self.jarvis._last_text_t < 3.0):
                txt_disp = self.jarvis.last_text[:46]
                tw_sp = cv2.getTextSize(txt_disp, FONT, 0.38, 1)[0][0]
                sp_x1 = 16; sp_y1 = fh - BOT_H - 42; sp_x2 = sp_x1 + tw_sp + 36; sp_y2 = sp_y1 + 28
                _glass_panel(out, sp_x1, sp_y1, sp_x2, sp_y2, col=(26, 24, 36), alpha=0.92, border_col=_PURPLE, r=8)
                ui_icons.draw_icon(out, "waveform", sp_x1 + 14, (sp_y1 + sp_y2) // 2, 14, _PURPLE, 1)
                cv2.putText(out, txt_disp, (sp_x1 + 26, sp_y1 + 18), FONT, 0.38, (220, 210, 255), 1, cv2.LINE_AA)

        # ── 9. BIN Zone (Interactive Trash Bin when pinching) ─────────────
        if self.pinch:
            bx1 = fw // 2 - 65; by1 = fh - 62
            bx2 = fw // 2 + 65; by2 = fh - 6
            in_bin = bx1 < cx < bx2 and by1 < cy < by2
            bc = _RED if in_bin else (45, 45, 140)
            _glass_panel(out, bx1, by1, bx2, by2, col=bc, alpha=0.90, border_col=_WHITE if in_bin else (80, 80, 200), r=12)
            ui_icons.draw_icon(out, "clear", fw // 2 - 24, by1 + 26, 18, _WHITE, 2)
            cv2.putText(out, "CLEAR", (fw // 2 - 4, by1 + 32), FONTB, 0.58, _WHITE, 2, cv2.LINE_AA)

        # ── 10. Character Bounding Boxes & Tags (OCR & Math) ──────────────
        if curr_tool in ("air_text", "smart_math") and self.ai_result and self.ai_result.get("characters"):
            box_col = (50, 220, 100) if curr_tool == "smart_math" else (240, 190, 0)
            for cr in self.ai_result["characters"]:
                bx, by, bw, bh = cr["box"]
                cv2.rectangle(out, (bx, by), (bx + bw, by + bh), box_col, 1, cv2.LINE_AA)
                tag_y = max(by - 22, BAR_H + 4)
                tag_w = max(bw, 22)
                _rrect(out, bx, tag_y, bx + tag_w, tag_y + 18, (30, 32, 40), r=4, border_col=box_col)
                cv2.putText(out, cr["char"], (bx + 4, tag_y + 14), FONTB, 0.42, box_col, 1, cv2.LINE_AA)

        # ── 11. Floating Modern AI Result Card (Center Top) ───────────────
        if curr_tool in ("air_text", "smart_math"):
            is_math = curr_tool == "smart_math"
            card_w  = 560
            card_x1 = (fw - card_w) // 2
            card_x2 = card_x1 + card_w
            card_y1 = BAR_H + 10
            card_y2 = card_y1 + 58
            card_border = (60, 215, 110) if is_math else (235, 180, 0)

            _glass_panel(out, card_x1, card_y1, card_x2, card_y2, col=(18, 20, 26), alpha=0.94, border_col=card_border, r=14)

            # Left badge chip
            title_text = "SMART MATH SOLVER" if is_math else "AIR-WRITING OCR"
            title_w = cv2.getTextSize(title_text, FONTB, 0.38, 1)[0][0]
            _rrect(out, card_x1 + 14, card_y1 + 8, card_x1 + 24 + title_w, card_y1 + 24,
                   (28, 42, 32) if is_math else (44, 38, 22), r=4, border_col=card_border)
            cv2.putText(out, title_text, (card_x1 + 18, card_y1 + 20), FONTB, 0.38, card_border, 1, cv2.LINE_AA)

            # Result content
            if self.ai_result and self.ai_result.get("text"):
                res_str = self.ai_result["text"]
                conf_pct = int(self.ai_result.get("confidence", 0.0) * 100)
                cv2.putText(out, res_str, (card_x1 + 16, card_y1 + 48), FONTB, 0.72, _WHITE, 2, cv2.LINE_AA)

                # Confidence badge
                conf_badge = f"{conf_pct}% AI Conf"
                cw = cv2.getTextSize(conf_badge, FONTB, 0.36, 1)[0][0]
                _rrect(out, card_x2 - cw - 26, card_y1 + 10, card_x2 - 12, card_y1 + 28, (24, 42, 30), r=6, border_col=_GREEN)
                cv2.putText(out, conf_badge, (card_x2 - cw - 20, card_y1 + 23), FONTB, 0.36, _GREEN, 1, cv2.LINE_AA)
                cv2.putText(out, "Voice: 'Paste' | 'Clear'", (card_x2 - 190, card_y1 + 48), FONT, 0.34, (200, 205, 230), 1, cv2.LINE_AA)
            else:
                prompt_hint = "Draw equation (e.g. 2 + 3 * 4)..." if is_math else "Write letters in air (e.g. HELLO)..."
                cv2.putText(out, prompt_hint, (card_x1 + 16, card_y1 + 46), FONT, 0.48, _GREY, 1, cv2.LINE_AA)
                cv2.putText(out, "Voice: 'Solve' (or pause)", (card_x2 - 180, card_y1 + 46), FONT, 0.34, _DIM, 1, cv2.LINE_AA)

        # ── 12. Finger Cursor ─────────────────────────────────────────────
        if hand_present:
            if self.is_selection_mode:
                # High-tech dual-ring selection reticle (Cyan radar)
                cv2.circle(out, (cx, cy), 18, (20, 22, 28), -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 18, _CYAN, 2, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 6, _WHITE, -1, cv2.LINE_AA)
                # Mini crosshairs
                cv2.line(out, (cx - 24, cy), (cx - 19, cy), _CYAN, 2, cv2.LINE_AA)
                cv2.line(out, (cx + 19, cy), (cx + 24, cy), _CYAN, 2, cv2.LINE_AA)
                cv2.line(out, (cx, cy - 24), (cx, cy - 19), _CYAN, 2, cv2.LINE_AA)
                cv2.line(out, (cx, cy + 19), (cx, cy + 24), _CYAN, 2, cv2.LINE_AA)
                cv2.putText(out, "SELECT", (cx + 16, cy - 12), FONTB, 0.36, _CYAN, 1, cv2.LINE_AA)
            else:
                col = self._color() if not self.eraser else _WHITE
                cv2.circle(out, (cx, cy), 12, (15, 15, 20), -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 9, col, -1, cv2.LINE_AA)
                cv2.circle(out, (cx, cy), 3, _WHITE, -1, cv2.LINE_AA)

        # ── 13. HUD Gesture Pill (Bottom Left) ────────────────────────────
        if self.show_hud:
            g_state = "DRAWING" if self.is_drawing else ("SELECT" if self.is_selection_mode else "READY")
            g_col   = _GREEN if self.is_drawing else (_CYAN if self.is_selection_mode else _GREY)
            _glass_panel(out, 14, fh - BOT_H - 32, 106, fh - BOT_H - 8, col=(20, 22, 28), alpha=0.85, border_col=_BORDER, r=6)
            cv2.circle(out, (26, fh - BOT_H - 20), 3, g_col, -1, cv2.LINE_AA)
            cv2.putText(out, g_state, (34, fh - BOT_H - 16), FONTB, 0.38, g_col, 1, cv2.LINE_AA)

        # ── 14. Floating Toast Notification ───────────────────────────────
        if self.toast and time.time() - self.toast_t < 2.0:
            tw2 = cv2.getTextSize(self.toast, FONTB, 0.65, 2)[0][0]
            tmx = (fw - tw2) // 2
            ty1 = BAR_H + 74 if curr_tool in ("air_text", "smart_math") else BAR_H + 16
            _rrect(out, tmx - 24, ty1, tmx + tw2 + 24, ty1 + 42, (24, 26, 34), r=12, border_col=_CYAN, border_thick=2)
            ui_icons.draw_icon(out, "check", tmx - 8, ty1 + 21, 14, _GREEN, 2)
            cv2.putText(out, self.toast, (tmx + 8, ty1 + 28), FONTB, 0.60, _WHITE, 2, cv2.LINE_AA)

        # ── 15. Debug Stats Panel ─────────────────────────────────────────
        if self.show_debug:
            try:
                import psutil
                proc = psutil.Process()
                cpu  = psutil.cpu_percent(interval=None)
                mem  = proc.memory_info().rss // (1024 * 1024)
            except Exception:
                cpu = mem = 0
            lines = [f"FPS: {fps}", f"Lat: {self.lat_ms:.1f}ms", f"CPU: {cpu}%", f"Mem: {mem}MB"]
            dw = 140; dx = fw - dw - 14; dy = BAR_H + 10
            _glass_panel(out, dx, dy, dx + dw, dy + len(lines) * 20 + 8, col=(20, 22, 28), alpha=0.92, border_col=_BORDER, r=8)
            for i, l in enumerate(lines):
                cv2.putText(out, l, (dx + 12, dy + 18 + i * 20), FONT, 0.38, _GREY, 1, cv2.LINE_AA)

        # ── 16. Floating Command Prompt Pill (Bottom Center) ──────────────
        hint_text = "Voice: 'Math Mode'  •  'Whiteboard'  •  'Save'  •  'Clear Canvas'  •  'Open Gemini'"
        hw = cv2.getTextSize(hint_text, FONT, 0.35, 1)[0][0]
        capsule_w = hw + 38
        cx1 = (fw - capsule_w) // 2
        cy1 = fh - BOT_H + 4
        cx2 = cx1 + capsule_w
        cy2 = fh - 6
        _glass_panel(out, cx1, cy1, cx2, cy2, col=(18, 20, 26), alpha=0.92, border_col=_BORDER, r=12)
        ui_icons.draw_icon(out, "mic", cx1 + 14, (cy1 + cy2) // 2, 12, _CYAN, 1)
        cv2.putText(out, hint_text, (cx1 + 26, cy1 + 16), FONT, 0.35, _WHITE, 1, cv2.LINE_AA)

        # ── 17. Gemini AI Chat Overlay ────────────────────────────────────
        if self.chat_mode:
            self._draw_chat_overlay(out, fw, fh)

        # ── 18. Manual Action Side Menu ───────────────────────────────────
        if self.menu_open:
            self._draw_side_menu(out, fw, fh)

    def _draw_mic_button(self, out: np.ndarray, fw: int, fh: int, hand_x: int, eff_mx: int = -1, eff_my: int = -1):
        """Draws an interactive microphone button indicating JARVIS state with modern glowing badges."""
        j_status = self.jarvis.status
        if eff_mx < 0:
            eff_mx, eff_my = getattr(self, 'mouse_pos', (-1, -1))

        if j_status == JarvisStatus.READY:
            j_label  = "JARVIS OFF"
            j_col    = _GREY
            btn_bg   = (28, 30, 38)
            border_c = _BORDER
        elif j_status == JarvisStatus.LISTENING:
            j_label  = "LISTENING..."
            j_col    = _CYAN
            btn_bg   = (16, 44, 52)
            border_c = _CYAN
        elif j_status == JarvisStatus.PROCESSING:
            j_label  = "PROCESSING..."
            j_col    = _ORANGE
            btn_bg   = (44, 34, 18)
            border_c = _ORANGE
        elif j_status == JarvisStatus.SPEAKING:
            j_label  = "SPEAKING..."
            j_col    = _PURPLE
            btn_bg   = (40, 22, 50)
            border_c = _PURPLE
        else:
            j_label  = "OFFLINE"
            j_col    = _DIM
            btn_bg   = (24, 24, 28)
            border_c = _BORDER

        tw = cv2.getTextSize(j_label, FONTB, 0.44, 1)[0][0]
        btn_w = tw + 46
        btn_h = 36
        btn_x2 = hand_x - 12
        btn_x1 = btn_x2 - btn_w
        btn_y1 = (BAR_H - btn_h) // 2
        btn_y2 = btn_y1 + btn_h

        self.mic_bbox = (btn_x1, btn_y1, btn_x2, btn_y2)
        is_hover = (btn_x1 <= eff_mx <= btn_x2 and btn_y1 <= eff_my <= btn_y2)
        if is_hover:
            border_c = _WHITE

        # Draw modern pill container
        _rrect(out, btn_x1, btn_y1, btn_x2, btn_y2, btn_bg, r=10, border_col=border_c)

        mic_cx = btn_x1 + 18
        mic_cy = BAR_H // 2

        # Pulsing soundwave animation when actively listening
        if j_status == JarvisStatus.LISTENING:
            pulse_r = int(14 + 3 * np.sin(time.time() * 8))
            cv2.circle(out, (mic_cx, mic_cy), pulse_r, _CYAN, 1, cv2.LINE_AA)
            ui_icons.draw_icon(out, "waveform", mic_cx, mic_cy, 16, _CYAN, 2)
        else:
            ui_icons.draw_icon(out, "mic", mic_cx, mic_cy, 16, j_col, 2)

        cv2.putText(out, j_label, (btn_x1 + 34, BAR_H // 2 + 5), FONTB, 0.44, j_col, 1, cv2.LINE_AA)

    # ── Finger Interaction Handler ────────────────────────────────────────────

    def _handle_finger_interaction(self, cx: int, cy: int, now: float) -> bool:
        """Process finger touch/hover on top bar swatches, pills, and buttons."""
        last_t = getattr(self, '_last_finger_select_time', 0.0)

        # 1. Color Swatches Hit Test (Finger Pointing / Selection)
        for (bx1, by1, bx2, by2), idx in self.swatch_bboxes:
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if self.color_idx != idx or self.eraser or (now - last_t > 0.6):
                    self.color_idx = idx
                    self.eraser = False
                    col_name = COLORS[idx][0]
                    self._toast(f"Color: {col_name}")
                    self._last_finger_select_time = now
                return True

        # 2. Tool Pill Hit Test (Cycle Tools)
        if self.tool_pill_bbox:
            bx1, by1, bx2, by2 = self.tool_pill_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 0.8:
                    if self.eraser:
                        self.eraser = False
                    else:
                        self.tool_idx = (self.tool_idx + 1) % len(TOOLS)
                    tool_name = TOOL_LABELS.get(self._tool(), "Pen")
                    self._toast(f"Tool: {tool_name}")
                    self._last_finger_select_time = now
                return True

        # 3. Whiteboard Toggle Pill
        if self.wb_pill_bbox:
            bx1, by1, bx2, by2 = self.wb_pill_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 0.8:
                    self.white_mode = not self.white_mode
                    self._toast("Whiteboard ON" if self.white_mode else "Camera Mode ON")
                    self._last_finger_select_time = now
                return True

        # 4. Brush Size Controls (+ / -)
        if self.brush_plus_bbox:
            bx1, by1, bx2, by2 = self.brush_plus_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 0.35:
                    self.brush = min(30, self.brush + 2)
                    self._toast(f"Brush: {self.brush}px")
                    self._last_finger_select_time = now
                return True

        if self.brush_minus_bbox:
            bx1, by1, bx2, by2 = self.brush_minus_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 0.35:
                    self.brush = max(2, self.brush - 2)
                    self._toast(f"Brush: {self.brush}px")
                    self._last_finger_select_time = now
                return True

        # 5. Hamburger Menu Button
        if self.menu_btn_bbox:
            bx1, by1, bx2, by2 = self.menu_btn_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 0.8:
                    self.menu_open = not self.menu_open
                    self._last_finger_select_time = now
                return True

        # 6. JARVIS Mic Button
        if self.mic_bbox and self.jarvis:
            bx1, by1, bx2, by2 = self.mic_bbox
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                if now - last_t > 1.2:
                    self.jarvis.listen_once_async()
                    self._last_finger_select_time = now
                return True

        # 7. Side Menu Action Items (if open)
        if self.menu_open and self.menu_panel_bbox:
            px1, py1, px2, py2 = self.menu_panel_bbox
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if now - last_t > 0.6:
                    for (bx1, by1, bx2, by2), action in self.menu_items:
                        if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                            action()
                            self.menu_open = False
                            self._last_finger_select_time = now
                            return True
        return False

    # ── Mouse Interaction Handler ─────────────────────────────────────────────

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse hover and clicks on all modern UI elements."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_pos = (x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pos = (x, y)

            # 1. Hamburger Menu Toggle Button
            if self.menu_btn_bbox:
                bx1, by1, bx2, by2 = self.menu_btn_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.menu_open = not self.menu_open
                    return

            # 2. Side Menu interactions (when open)
            if self.menu_open:
                if self.menu_panel_bbox:
                    px1, py1, px2, py2 = self.menu_panel_bbox
                    if px1 <= x <= px2 and py1 <= y <= py2:
                        for (bx1, by1, bx2, by2), action in self.menu_items:
                            if bx1 <= x <= bx2 and by1 <= y <= by2:
                                log.info("Side menu item clicked.")
                                action()
                                self.menu_open = False
                                return
                        return
                    else:
                        # Clicked outside panel: dismiss drawer
                        self.menu_open = False
                        return

            # 3. JARVIS Mic Button Click
            if self.mic_bbox and self.jarvis:
                bx1, by1, bx2, by2 = self.mic_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    log.info("Mic button clicked.")
                    self.jarvis.listen_once_async()
                    return

            # 4. Color Swatches Click
            for (bx1, by1, bx2, by2), idx in self.swatch_bboxes:
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.color_idx = idx
                    self.eraser = False
                    col_name = COLORS[idx][0]
                    self._toast(f"Color: {col_name}")
                    return

            # 5. Tool Pill Click (Cycle Tools)
            if self.tool_pill_bbox:
                bx1, by1, bx2, by2 = self.tool_pill_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    if self.eraser:
                        self.eraser = False
                    else:
                        self.tool_idx = (self.tool_idx + 1) % len(TOOLS)
                    tool_name = TOOL_LABELS.get(self._tool(), "Pen")
                    self._toast(f"Tool: {tool_name}")
                    return

            # 6. Whiteboard Toggle Pill Click
            if self.wb_pill_bbox:
                bx1, by1, bx2, by2 = self.wb_pill_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.white_mode = not self.white_mode
                    self._toast("Whiteboard ON" if self.white_mode else "Camera Mode ON")
                    return

            # 7. Brush Size Controls Click
            if self.brush_plus_bbox:
                bx1, by1, bx2, by2 = self.brush_plus_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.brush = min(30, self.brush + 2)
                    self._toast(f"Brush: {self.brush}px")
                    return

            if self.brush_minus_bbox:
                bx1, by1, bx2, by2 = self.brush_minus_bbox
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.brush = max(2, self.brush - 2)
                    self._toast(f"Brush: {self.brush}px")
                    return

    # ── Modern Side Drawer Command Center ─────────────────────────────────────

    def _draw_side_menu(self, out: np.ndarray, fw: int, fh: int):
        """Draws the modern floating frosted glass command center."""
        panel_w = 330
        panel_x1 = 12
        panel_y1 = BAR_H + 10
        panel_x2 = panel_x1 + panel_w
        panel_y2 = fh - BOT_H - 10
        self.menu_panel_bbox = (panel_x1, panel_y1, panel_x2, panel_y2)

        # Smooth dimmed backdrop overlay over canvas
        roi = out[BAR_H:fh - BOT_H, 0:fw]
        roi[:] = (roi * 0.40).astype(np.uint8)

        # Floating glass drawer container
        _glass_panel(out, panel_x1, panel_y1, panel_x2, panel_y2, col=(18, 20, 26), alpha=0.96, border_col=(65, 70, 85), r=16)

        # Drawer Header
        cv2.putText(out, "COMMAND CENTER", (panel_x1 + 18, panel_y1 + 28), FONTB, 0.52, _WHITE, 2, cv2.LINE_AA)
        cv2.circle(out, (panel_x1 + 215, panel_y1 + 22), 3, _CYAN, -1, cv2.LINE_AA)

        # Close button in header
        close_bx1, close_by1 = panel_x2 - 34, panel_y1 + 12
        close_bx2, close_by2 = close_bx1 + 22, close_by1 + 22
        _rrect(out, close_bx1, close_by1, close_bx2, close_by2, (30, 32, 40), r=6, border_col=_BORDER)
        ui_icons.draw_icon(out, "close", (close_bx1 + close_bx2) // 2, (close_by1 + close_by2) // 2, 12, _GREY, 2)
        self.menu_items.clear()
        self.menu_items.append(((close_bx1, close_by1, close_bx2, close_by2), lambda: None)) # Closes menu

        if not self.controller:
            return
        c = self.controller

        menu_structure = [
            {"type": "header", "label": "CANVAS & EXPORT"},
            {"type": "action", "label": "Clear Canvas", "icon": "clear", "key": "C", "action": lambda: c.clear_canvas()},
            {"type": "action", "label": "Save Canvas", "icon": "save", "key": "S", "action": lambda: c.save_canvas()},
            {"type": "header", "label": "DRAWING TOOLS"},
            {"type": "action", "label": "Pen (Normal)", "icon": "pen", "key": "1", "action": lambda: c.select_tool("pen")},
            {"type": "action", "label": "Marker", "icon": "marker", "key": "2", "action": lambda: c.select_tool("marker")},
            {"type": "action", "label": "Highlighter", "icon": "highlighter", "key": "3", "action": lambda: c.select_tool("highlighter")},
            {"type": "action", "label": "Shape Pen", "icon": "shape_pen", "key": "4", "action": lambda: c.select_tool("shape_pen")},
            {"type": "action", "label": "Eraser", "icon": "eraser", "key": "E", "action": lambda: c.select_tool("eraser")},
            {"type": "header", "label": "AI RECOGNITION"},
            {"type": "action", "label": "Smart Math Solver", "icon": "smart_math", "key": "M", "action": lambda: c.select_tool("smart_math")},
            {"type": "action", "label": "Air-Writing OCR", "icon": "air_text", "key": "A", "action": lambda: c.select_tool("air_text")},
            {"type": "action", "label": "Gemini AI Chat", "icon": "gemini", "key": "G", "action": lambda: c.open_gemini()},
            {"type": "header", "label": "VIEW & BRUSH"},
            {"type": "action", "label": "Toggle Whiteboard", "icon": "whiteboard", "key": "W", "action": lambda: c.toggle_whiteboard()},
            {"type": "action", "label": "Increase Brush", "icon": "increase_brush", "key": "+", "action": lambda: c.increase_brush()},
            {"type": "action", "label": "Decrease Brush", "icon": "decrease_brush", "key": "-", "action": lambda: c.decrease_brush()},
        ]

        cy = panel_y1 + 44
        pad_x = panel_x1 + 14
        item_w = panel_w - 28
        item_h = 32
        mx, my = getattr(self, 'mouse_pos', (-1, -1))

        for item in menu_structure:
            if item["type"] == "header":
                cy += 6
                cv2.putText(out, item["label"], (pad_x + 4, cy + 10), FONTB, 0.36, _CYAN, 1, cv2.LINE_AA)
                cv2.line(out, (pad_x, cy + 16), (pad_x + item_w, cy + 16), (36, 40, 50), 1, cv2.LINE_AA)
                cy += 24
            elif item["type"] == "action":
                bx1 = pad_x
                by1 = cy
                bx2 = pad_x + item_w
                by2 = cy + item_h

                is_item_hover = (bx1 <= mx <= bx2 and by1 <= my <= by2)

                # Active tool check
                label = item["label"]
                current_tool = self._tool()
                is_active = False
                if label == "Pen (Normal)" and current_tool == "pen" and not self.eraser: is_active = True
                elif label == "Marker" and current_tool == "marker" and not self.eraser: is_active = True
                elif label == "Highlighter" and current_tool == "highlighter" and not self.eraser: is_active = True
                elif label == "Shape Pen" and current_tool == "shape_pen" and not self.eraser: is_active = True
                elif label == "Eraser" and self.eraser: is_active = True
                elif label == "Smart Math Solver" and current_tool == "smart_math" and not self.eraser: is_active = True
                elif label == "Air-Writing OCR" and current_tool == "air_text" and not self.eraser: is_active = True
                elif label == "Gemini AI Chat" and self.chat_mode: is_active = True
                elif label == "Toggle Whiteboard" and self.white_mode: is_active = True

                # Card styling
                if is_active:
                    bg_col = (26, 44, 48); border_col = _CYAN; tcol = _WHITE
                elif is_item_hover:
                    bg_col = (42, 46, 58); border_col = _BORDER_LGT; tcol = _WHITE
                else:
                    bg_col = (28, 30, 38); border_col = _BORDER; tcol = (200, 205, 215)

                _rrect(out, bx1, by1, bx2, by2, bg_col, r=7, border_col=border_col)

                # Icon container box
                _rrect(out, bx1 + 3, by1 + 3, bx1 + 27, by2 - 3, (18, 20, 26), r=5)
                ui_icons.draw_icon(out, item["icon"], bx1 + 15, (by1 + by2) // 2, 14, _CYAN if is_active else _GREY, 2)

                # Label text
                cv2.putText(out, label, (bx1 + 36, by1 + 20), FONT, 0.44, tcol, 1, cv2.LINE_AA)

                # Keyboard shortcut tag
                key_tag = item.get("key", "")
                if key_tag:
                    kw = cv2.getTextSize(key_tag, FONTB, 0.35, 1)[0][0]
                    kx2 = bx2 - 8; kx1 = kx2 - max(18, kw + 8)
                    ky1 = by1 + 6; ky2 = by2 - 6
                    _rrect(out, kx1, ky1, kx2, ky2, (18, 20, 26), r=4, border_col=(45, 48, 60))
                    cv2.putText(out, key_tag, (kx1 + 4, by1 + 19), FONTB, 0.35, _CYAN if is_active else _DIM, 1, cv2.LINE_AA)

                self.menu_items.append(((bx1, by1, bx2, by2), item["action"]))
                cy += item_h + 5

    # ── Keyboard handler ──────────────────────────────────────────────────────

    def _handle_key(self, k: int) -> bool:
        if k == 255: return True

        # ── Chat overlay intercepts ALL keys when active ────────────────────────
        if self.chat_mode:
            return self._handle_chat_key(k)

        # ESC or Q: if in Smart Math or Air-Text, exit back to normal Pen mode
        if k in (27, ord('q'), ord('Q')):
            if self._tool() in ("air_text", "smart_math"):
                self.tool_idx  = TOOLS.index("pen")
                self.ai_result = None
                self.ai_solved = False
                self._toast("Exited to Pen")
                return True
            if k == 27:
                return False   # ESC exits app only when in normal mode

        # ── JARVIS Push-to-Talk (V key) ────────────────────────────────────
        if k in (ord('v'), ord('V')):
            if self.jarvis:
                self.jarvis.toggle_listening()
            return True

        # ── Gemini AI Chat (G key) ───────────────────────────────────────
        if k in (ord('g'), ord('G')):
            self.chat_mode    = not self.chat_mode
            self.chat_input   = ""
            self.chat_answer  = ""
            self.chat_stamped = False
            self.chat_api_input = False
            self._toast("Gemini Chat ON" if self.chat_mode else "Gemini Chat OFF")
            return True

        # Color 1-7
        if ord('1') <= k <= ord('7'):
            self.color_idx = k - ord('1')
            self.eraser    = False
            self._toast(f"Color: {COLORS[self.color_idx][0]}")

        elif k in (ord('t'), ord('T')):
            self.tool_idx = (self.tool_idx + 1) % len(TOOLS)
            self.eraser   = False
            self.ai_result = None
            self.ai_solved = False
            self._toast(f"Tool: {TOOL_LABELS[TOOLS[self.tool_idx]]}")

        elif k in (ord('x'), ord('X')):
            if self._tool() == "air_text":
                # Already in Air-Text — toggle back to pen
                self.tool_idx  = TOOLS.index("pen")
                self.ai_result = None
                self.ai_solved = False
                self._toast("Air-Text OFF → Pen")
            else:
                self.tool_idx = TOOLS.index("air_text")
                self.eraser   = False
                self.ai_result = None
                self.ai_solved = False
                self._toast("Air-Writing Mode [TXT]")

        elif k in (ord('m'), ord('M')):
            if self._tool() == "smart_math":
                # Already in Smart Math — toggle back to pen
                self.tool_idx  = TOOLS.index("pen")
                self.ai_result = None
                self.ai_solved = False
                self._toast("Smart Math OFF → Pen")
            else:
                self.tool_idx = TOOLS.index("smart_math")
                self.eraser   = False
                self.ai_result = None
                self.ai_solved = False
                self._toast("Smart Math Mode [MTH]")

        elif k in (13, 10, 32):  # Enter, Return, Space
            if self._tool() in ("air_text", "smart_math"):
                if self.ai_result and self.ai_result.get("text"):
                    self._stamp_ai_result()
                else:
                    self._run_ai_solve()
            else:
                self._run_ai_solve()

        elif k in (ord('e'), ord('E')):
            self.eraser = not self.eraser
            self._toast("Eraser ON" if self.eraser else "Pen mode")

        elif k in (ord('w'), ord('W')):
            self.white_mode = not self.white_mode
            self._toast("Whiteboard ON" if self.white_mode else "Camera mode")

        elif k in (ord('c'), ord('C')):
            if self.canvas: self.canvas.clear()
            self.ai_result = None
            self.ai_solved = False
            self.last_draw_time = 0.0
            self._toast("Canvas cleared")


        elif k in (ord('+'), ord('=')):
            self.brush = min(self.brush + 2, 40)
            self._toast(f"Brush: {self.brush}")

        elif k == ord('-'):
            self.brush = max(self.brush - 2, 2)
            self._toast(f"Brush: {self.brush}")

        elif k in (ord('s'), ord('S')):
            self._save()

        elif k in (ord('d'), ord('D')):
            self.show_debug = not self.show_debug

        elif k in (ord('h'), ord('H')):
            self.show_hud = not self.show_hud

        return True

    # ── Gemini Chat Overlay ─────────────────────────────────────────────────────

    def _draw_chat_overlay(self, out: np.ndarray, fw: int, fh: int):
        """Render the Gemini AI chat panel over the frame."""
        # Panel geometry
        pw   = min(fw - 40, 900)
        ph   = 260
        px   = (fw - pw) // 2
        py   = fh - BOT_H - ph - 12

        # Semi-transparent dark background
        overlay = out.copy()
        cv2.rectangle(overlay, (px, py), (px+pw, py+ph), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.92, out, 0.08, 0, out)
        cv2.rectangle(out, (px, py), (px+pw, py+ph), (80, 60, 200), 2)  # purple border

        # Header
        _rrect(out, px, py, px+pw, py+34, (35, 25, 60), r=0)
        cv2.putText(out, "GEMINI AI CHAT (VOICE CONTROLLED)",
                    (px+12, py+22), FONTB, 0.55, (200, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(out, "Say: 'Close Gemini'",
                    (px+pw-155, py+22), FONT, 0.35, _CYAN, 1, cv2.LINE_AA)

        # API key prompt if no key set
        if not self.gemini.has_key:
            if self.chat_api_input or self.chat_input:
                prompt_txt = "API Key: " + self.chat_input + "▋"
                cv2.putText(out, prompt_txt, (px+16, py+80),
                            FONTB, 0.55, (255, 200, 80), 1, cv2.LINE_AA)
                cv2.putText(out, "Press Enter to save key | ESC to cancel",
                            (px+16, py+112), FONT, 0.38, _DIM, 1, cv2.LINE_AA)
            else:
                cv2.putText(out, "No API key set.", (px+16, py+75),
                            FONTB, 0.6, (60, 80, 255), 1, cv2.LINE_AA)
                cv2.putText(out, "Get free key: aistudio.google.com/app/apikey",
                            (px+16, py+105), FONT, 0.42, _GREY, 1, cv2.LINE_AA)
                cv2.putText(out, "Type your API key and press Enter",
                            (px+16, py+135), FONT, 0.42, (100, 200, 255), 1, cv2.LINE_AA)
            return

        # Input box
        inp_y = py + 70
        _rrect(out, px+10, inp_y-24, px+pw-10, inp_y+10, (30, 28, 45), r=6)
        cv2.rectangle(out, (px+10, inp_y-24), (px+pw-10, inp_y+10), (80, 60, 160), 1)
        disp_input = self.chat_input[-72:] if len(self.chat_input) > 72 else self.chat_input
        cursor = "▋" if not self.chat_thinking else ""
        cv2.putText(out, "Search: " + disp_input + cursor,
                    (px+18, inp_y), FONTB, 0.50, (220, 200, 255), 1, cv2.LINE_AA)

        sep_y = inp_y + 20
        cv2.line(out, (px+10, sep_y), (px+pw-10, sep_y), (60, 50, 90), 1)

        # Answer / status area
        ans_y = sep_y + 30
        if self.chat_thinking:
            dots = "." * (int(time.time() * 3) % 4)
            cv2.putText(out, f"Gemini is thinking{dots}",
                        (px+16, ans_y), FONTB, 0.55, (180, 120, 255), 1, cv2.LINE_AA)
        elif self.chat_answer:
            # Word-wrap answer into lines of ~90 chars
            words  = self.chat_answer.split()
            lines  = []
            line   = ""
            for w in words:
                if len(line) + len(w) + 1 > 88:
                    lines.append(line)
                    line = w
                else:
                    line = (line + " " + w).strip()
            if line:
                lines.append(line)

            for i, ln in enumerate(lines[:3]):
                cv2.putText(out, ln, (px+16, ans_y + i*26),
                            FONT, 0.44, _WHITE, 1, cv2.LINE_AA)

            # Action hint
            hint_y = py + ph - 28
            if self.chat_stamped:
                cv2.putText(out, "Stamped on canvas!  Say 'Close Gemini' to return, or ask a new question.",
                            (px+16, hint_y), FONT, 0.40, (50, 220, 100), 1, cv2.LINE_AA)
            else:
                cv2.putText(out, "Voice: Say 'Paste into screen' to stamp in handwritten format  |  Say 'Close Gemini'",
                            (px+16, hint_y), FONT, 0.38, (120, 220, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(out, "Say: 'Search who is the prime minister of india' (or speak any question)...",
                        (px+16, ans_y), FONT, 0.46, (170, 170, 210), 1, cv2.LINE_AA)

    def _handle_chat_key(self, k: int) -> bool:
        """
        Handle keyboard input while chat overlay is open.
        ONLY Enter and ESC are control keys:
          - ESC: Close the chat overlay
          - Enter: Send question to Gemini (or stamp answer onto canvas)
          - Backspace: Delete previous character
          - All other keys (letters a-z, numbers, symbols): Type into the question box
        """
        # ESC = close chat overlay
        if k == 27:
            if self.chat_api_input:
                self.chat_api_input = False
                self.chat_input     = ""
            else:
                self.chat_mode = False
            return True

        # Backspace = delete last character
        if k in (8, 127):
            self.chat_input = self.chat_input[:-1]
            return True

        # Enter = send question OR stamp answer
        if k in (13, 10):
            if not self.gemini.has_key or self.chat_api_input:
                candidate = self.chat_input.strip()
                if candidate:
                    if GeminiChatEngine.is_valid_key(candidate):
                        if self.gemini.set_api_key(candidate):
                            self._toast("Gemini API Key saved!")
                            self.chat_api_input = False
                            self.chat_input     = ""
                            return True
                    else:
                        self._toast("Invalid key format. Please enter a valid Gemini API key.")
                        return True
                self.chat_api_input = False
                self.chat_input     = ""
                return True

            current_input = self.chat_input.strip()

            # If user entered a new/different question, send it to Gemini
            if current_input and current_input != getattr(self, "_last_chat_query", ""):
                if not self.chat_thinking:
                    self._last_chat_query = current_input
                    self.chat_answer   = ""
                    self.chat_stamped  = False
                    self.chat_thinking = True
                    self.gemini.ask(current_input, self._gemini_callback)
                return True

            # If an answer is ready and hasn't been stamped yet, stamp it onto canvas
            if self.chat_answer and not self.chat_stamped:
                self._stamp_gemini_answer()
                return True

            # If input is present and idle, send question to Gemini
            if current_input and not self.chat_thinking:
                self._last_chat_query = current_input
                self.chat_answer   = ""
                self.chat_stamped  = False
                self.chat_thinking = True
                self.gemini.ask(current_input, self._gemini_callback)
            return True

        # All printable characters (ASCII 32 to 126, including 'c', 'g', 'k', space, etc.)
        # type directly into the question input. NO special key shortcuts intercept here!
        if 32 <= k <= 126 and len(self.chat_input) < 200:
            if not self.gemini.has_key:
                self.chat_api_input = True
            self.chat_input += chr(k)

        return True

    def _gemini_callback(self, answer: str, error):
        """Called from Gemini background thread — only updates state, never touches OpenCV."""
        self.chat_thinking = False
        self.chat_answer   = answer
        if error:
            log.warning(f"Gemini error: {error}")
        elif answer and self.jarvis:
            first_sent = answer.split(".")[0].strip()
            if first_sent:
                self.jarvis.speak(first_sent)

    def _stamp_gemini_answer(self):
        """Stamp the Gemini answer onto the canvas in handwritten font + draw color."""
        if not self.canvas or not self.chat_answer:
            return
        self.canvas.save_snapshot()
        color = self._color()
        # Position in visible drawing area
        x, y = 80, 100
        self._draw_handwritten_text(self.canvas.surface, self.chat_answer, x, y, color, font_size=52)
        self.chat_stamped = True
        self.chat_input   = ""
        self._toast("Gemini answer stamped in your color!")
        log.info("Gemini answer stamped to canvas")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _color(self):
        return COLORS[self.color_idx][1]

    def _tool(self):
        return TOOLS[self.tool_idx]

    def _draw_handwritten_text(self, surface: np.ndarray, text: str,
                               x: int, y: int, color_bgr: tuple, font_size: int = 64):
        """
        Render `text` onto `surface` at (x, y) in a handwritten font style,
        using the given BGR color. Uses PIL + Caveat-Regular.ttf when available,
        falls back to OpenCV FONT_HERSHEY_SCRIPT_SIMPLEX otherwise.
        """
        # Convert BGR → RGB for PIL
        r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]

        import textwrap
        # Word-wrap text if line exceeds ~38 characters
        if len(text) > 38 and "\n" not in text:
            text = "\n".join(textwrap.wrap(text, width=38))

        # ── Try PIL + Caveat font ──────────────────────────────────────────
        font_path = "data/fonts/Caveat-Regular.ttf"
        try:
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(cv2.cvtColor(surface, cv2.COLOR_BGR2RGB))
            draw    = ImageDraw.Draw(pil_img)
            font    = ImageFont.truetype(font_path, size=font_size)

            # Measure text for background pill and bounds clamping
            bbox = draw.textbbox((x, y), text, font=font, spacing=6)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # Clamp coordinates so text stays on-screen
            if x + tw > W - 40:
                x = max(20, W - 40 - tw)
            if y + th > H - 70:
                y = max(60, H - 70 - th)

            bbox = draw.textbbox((x, y), text, font=font, spacing=6)
            pad  = 12
            # Semi-transparent dark pill so text is legible on any background
            overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)
            ov_draw.rounded_rectangle(
                (bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad),
                radius=12, fill=(18, 18, 24, 180))
            pil_img = Image.alpha_composite(
                pil_img.convert("RGBA"), overlay).convert("RGB")

            # Draw the handwritten text in current draw color
            draw2 = ImageDraw.Draw(pil_img)
            # Slight shadow for depth
            draw2.text((x+2, y+2), text, font=font, fill=(0, 0, 0, 140), spacing=6)
            draw2.text((x, y),     text, font=font, fill=(r, g, b), spacing=6)

            result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            np.copyto(surface, result)
            log.debug(f"Handwritten stamp: '{text}' at ({x},{y}) color=({r},{g},{b})")
            return

        except Exception as e:
            log.debug(f"PIL handwritten render failed ({e}), using OpenCV fallback.")

        # ── Fallback: OpenCV script font ──────────────────────────────────
        scale = font_size / 48.0
        thickness = max(2, int(scale * 2))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, thickness)
        # Dark pill background
        pad = 8
        cv2.rectangle(surface,
                      (x - pad, y - th - pad),
                      (x + tw + pad, y + pad),
                      (20, 20, 25), -1)
        # Shadow
        cv2.putText(surface, text, (x+2, y+2),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, (0, 0, 0), thickness+1, cv2.LINE_AA)
        # Colored text
        cv2.putText(surface, text, (x, y),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, scale, color_bgr, thickness, cv2.LINE_AA)

    def _reset_draw(self):
        self.prev_x = self.prev_y = None
        self.pts        = []
        self.is_drawing = False

    def _toast(self, msg: str):
        self.toast   = msg
        self.toast_t = time.time()

    def _shape_detect(self):
        if len(self.pts) < 25:
            return
        shape, approx = detect_shape(self.pts)
        if shape == SHAPE_UNKNOWN:
            return
        DrawingEngine.erase_points(self.canvas.surface, self.pts, self.brush)
        contour = np.array(self.pts, dtype=np.int32).reshape((-1, 1, 2))
        draw_perfect_shape(self.canvas.surface, shape, approx, contour, self._color())
        self._toast(f"Shape: {shape}")

    def _save(self):
        from datetime import datetime
        ts   = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = f"exports/AeroScript_{ts}.png"
        try:
            white = np.ones_like(self.canvas.surface) * 255
            img   = self.canvas.composite(white)
            cv2.imwrite(path, img)
            self._toast(f"Saved: AeroScript_{ts}.png")
            log.info(f"Exported {path}")
        except Exception as ex:
            self._toast("Save failed!")
            log.error(f"Export error: {ex}")

    # ── JARVIS Initialisation ────────────────────────────────────────────────

    def _init_jarvis(self):
        """Initialize JARVIS voice assistant. Failure is non-fatal."""
        if not _JARVIS_AVAILABLE:
            log.warning("JARVIS module not importable — voice control disabled.")
            return
        try:
            cfg = _cfg
            self.jarvis = JarvisAssistant(
                startup_speech    = cfg.JARVIS_STARTUP_SPEECH,
                wake_word_enabled = cfg.JARVIS_WAKE_WORD_ENABLED,
                wake_words        = cfg.JARVIS_WAKE_WORDS,
                energy_threshold  = cfg.JARVIS_STT_ENERGY_THRESHOLD,
                listen_timeout    = cfg.JARVIS_LISTEN_TIMEOUT,
                phrase_timeout    = cfg.JARVIS_PHRASE_TIMEOUT,
                tts_rate          = cfg.JARVIS_TTS_RATE,
                tts_volume        = cfg.JARVIS_TTS_VOLUME,
                response_cooldown = cfg.JARVIS_RESPONSE_COOLDOWN_S,
            )
            # Controller is already instantiated in __init__, just bind it to jarvis
            self.jarvis.bind_app(self)
            ok = self.jarvis.start()
            if ok:
                log.info("JARVIS: OK")
            else:
                log.warning("JARVIS: started with reduced capability (no microphone).")
        except Exception as e:
            log.error(f"JARVIS init failed: {e}")
            self.jarvis     = None
            self.controller = None

    # ── Startup screen ────────────────────────────────────────────────────────

    def _startup(self) -> str:
        items = [
            ("canvas",       "Smart AI Canvas",      "Air-Writing, Math, Drawing, Shapes"),
            ("exit",         "Exit",                 "Quit AeroScript"),
        ]
        sel = 0
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, W, H)

        start_time = time.time()
        auto_duration = 1.0  # 1 second auto-launch without pressing Enter

        while True:
            elapsed = time.time() - start_time
            if elapsed >= auto_duration:
                return "canvas"

            f = np.zeros((H, W, 3), dtype=np.uint8)
            for y in range(H):
                v = int(16 + 16*(y/H))
                f[y, :] = (v, v, v+8)

            # Card
            cw, ch = 620, 420
            cx1 = (W-cw)//2; cy1 = (H-ch)//2
            cx2 = cx1+cw;    cy2 = cy1+ch
            _rrect(f, cx1, cy1, cx2, cy2, (40, 40, 46), r=18)
            cv2.line(f, (cx1+28, cy1), (cx2-28, cy1), _CYAN, 2)

            # Title
            cv2.putText(f, "AeroScript 2.0", (cx1+30, cy1+54),
                        FONTB, 1.15, _CYAN, 2, cv2.LINE_AA)
            cv2.putText(f, "AI-Powered Touchless Human-Computer Interaction",
                        (cx1+30, cy1+82), FONT, 0.44, _GREY, 1, cv2.LINE_AA)
            cv2.line(f, (cx1+28, cy1+98), (cx2-28, cy1+98), _SEP, 1)

            # Feature list highlighting voice-first & AI
            features = [
                ("JARVIS Click-to-Talk -- Click Mic icon or press 'V' to speak", (50, 220, 100)),
                ("  'Open Gemini' -> 'Search <question>' -> 'Paste into screen'", (200, 180, 255)),
                ("  'Open Math Mode' -> write math in the air -> 'Paste to screen'", (240, 200, 0)),
                ("  'Open Whiteboard' | 'Camera Mode' | 'Close AeroScript'", _WHITE),
                ("Touchless Air Gestures: Point index to draw, Pinch to pan/clear", _DIM),
            ]
            for i, (feat, col) in enumerate(features):
                cv2.putText(f, feat, (cx1+30, cy1+126+i*28),
                            FONT, 0.40, col, 1, cv2.LINE_AA)

            # Auto-start progress bar
            pct = min(1.0, elapsed / auto_duration)
            pw = int((cw - 60) * pct)
            _rrect(f, cx1+30, cy1+286, cx2-30, cy1+302, (28, 28, 34), r=6)
            if pw > 4:
                _rrect(f, cx1+30, cy1+286, cx1+30+pw, cy1+302, (0, 200, 240), r=6)
            cv2.putText(f, f"Starting automatically... ({int(pct*100)}%)",
                        (cx1+30, cy1+324), FONTB, 0.44, _WHITE, 1, cv2.LINE_AA)
            cv2.putText(f, "Hands-free ready -- Click JARVIS Mic to issue commands",
                        (cx1+30, cy1+348), FONT, 0.38, (50, 220, 100), 1, cv2.LINE_AA)

            cv2.putText(f, "Auto-launching... (Press ESC to cancel)",
                        (W//2-130, H-18), FONT, 0.38, _DIM, 1, cv2.LINE_AA)

            cv2.imshow(WIN, f)
            k = cv2.waitKey(30) & 0xFF
            if k == 27 or k == ord('q'):
                return "exit"
            if k in (13, 10):
                return "canvas"
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                return "exit"

    def _err(self, msg):
        f = np.zeros((360, 640, 3), dtype=np.uint8)
        for i, l in enumerate(msg.split("\n")):
            cv2.putText(f, l, (30, 80+i*50), FONT, 0.8, (80, 80, 200), 2)
        cv2.imshow(WIN, f)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    App().run()
