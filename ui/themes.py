"""
AeroScript 2.0 — UI Themes
Apple-inspired design tokens for the OpenCV overlay UI.
All color/size constants are defined here; never hardcode them in UI modules.
"""

# ─── Color Palette (BGR for OpenCV) ──────────────────────────────────────────

# Backgrounds & Panels
BG_DARK        = (30,  30,  30)    # #1E1E1E — main dark background
BG_PANEL       = (45,  45,  45)    # #2D2D2D — panel surface
BG_PANEL_LIGHT = (62,  62,  62)    # #3E3E3E — lighter panel / hover
BG_SELECTED    = (80,  80,  80)    # #505050 — selected item

# Accent Colors
ACCENT_CYAN     = (200, 180,  0)   # #00B4C8 — primary accent (BGR)
ACCENT_GREEN    = ( 50, 205,  50)  # #32CD32 — positive / success
ACCENT_ORANGE   = (  0, 140, 255)  # #FF8C00 — warning
ACCENT_RED      = ( 40,  40, 220)  # #DC2828 — danger / error

# Text
TEXT_PRIMARY   = (240, 240, 240)   # near-white
TEXT_SECONDARY = (160, 160, 160)   # grey
TEXT_DISABLED  = ( 90,  90,  90)

# Dividers
SEPARATOR      = ( 70,  70,  70)

# Tool icon states
ICON_ACTIVE    = ACCENT_CYAN
ICON_INACTIVE  = TEXT_SECONDARY
ICON_BG_ACTIVE = BG_SELECTED

# ─── Typography Equivalents ───────────────────────────────────────────────────
# OpenCV doesn't support custom fonts, we use FONT_HERSHEY variants

FONT         = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD    = 1   # cv2.FONT_HERSHEY_DUPLEX   (heavier weight)
FONT_MONO    = 4   # cv2.FONT_HERSHEY_PLAIN    (monospace-ish)

# Scale presets
FONT_SMALL   = 0.42
FONT_MEDIUM  = 0.58
FONT_LARGE   = 0.80
FONT_XLARGE  = 1.10

FONT_THICK_THIN   = 1
FONT_THICK_NORMAL = 2
FONT_THICK_BOLD   = 3

# ─── Layout Dimensions ────────────────────────────────────────────────────────

TOOLBAR_H       = 80    # px — top toolbar height
SIDE_W          = 80    # px — right side panel width
CORNER_RADIUS   = 10    # px — rounded rect radius (approximated with polygon)
ICON_SIZE       = 34    # px — tool icon bounding box
ICON_PADDING    = 10    # px — padding around icons in panel
STATUS_BAR_H    = 28    # px — bottom status bar height

# ─── HUD Dimensions ───────────────────────────────────────────────────────────

HUD_MARGIN      = 12    # px from edge
HUD_LINE_H      = 22    # px between HUD lines
HUD_BOX_ALPHA   = 0.65  # transparency of HUD background boxes

# ─── Color Swatches in Toolbar ───────────────────────────────────────────────

SWATCH_SIZE     = 26    # px — color swatch circle diameter
SWATCH_MARGIN   = 8     # px — gap between swatches


def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=CORNER_RADIUS,
                      filled=True, thickness=1, border_color=None):
    """
    Draw a smooth anti-aliased rounded rectangle on an OpenCV image.
    Supports optional subtle border color.
    """
    import cv2
    radius = max(2, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    AA = cv2.LINE_AA

    if filled:
        # Four corner circles with AA
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, AA)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, AA)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, AA)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, AA)

        # Three fill rectangles
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x1 + radius, y2 - radius), color, -1)
        cv2.rectangle(img, (x2 - radius, y1 + radius), (x2, y2 - radius), color, -1)

    if border_color is not None or not filled:
        b_col = border_color if border_color is not None else color
        b_thick = thickness if thickness > 0 else 1
        # Corner arcs
        cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, b_col, b_thick, AA)
        cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, b_col, b_thick, AA)
        cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, b_col, b_thick, AA)
        cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, b_col, b_thick, AA)
        # Edge lines
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), b_col, b_thick, AA)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), b_col, b_thick, AA)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), b_col, b_thick, AA)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), b_col, b_thick, AA)


def overlay_alpha(frame, panel, x, y, alpha=0.82):
    """
    Blend a solid panel over the frame at position (x, y) with given alpha.
    """
    import cv2, numpy as np
    h, w = panel.shape[:2]
    fh, fw = frame.shape[:2]
    h = min(h, fh - y)
    w = min(w, fw - x)
    if h <= 0 or w <= 0:
        return
    roi = frame[y:y+h, x:x+w]
    cv2.addWeighted(panel[:h, :w], alpha, roi, 1.0 - alpha, 0, roi)
    frame[y:y+h, x:x+w] = roi


def draw_glass_panel(frame, x1, y1, x2, y2, bg_color=BG_DARK, alpha=0.85, border_color=(65, 68, 76), radius=CORNER_RADIUS):
    """
    Draws a modern frosted glass container with smooth rounded corners and subtle highlight border.
    """
    import cv2, numpy as np
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return

    # In-place patch blend
    roi = frame[y1:y2, x1:x2]
    patch = np.full((h, w, 3), bg_color, dtype=np.uint8)
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, 0, roi)
    frame[y1:y2, x1:x2] = roi

    # Draw rounded border overlay
    if border_color is not None:
        draw_rounded_rect(frame, x1, y1, x2, y2, bg_color, radius=radius, filled=False, thickness=1, border_color=border_color)

