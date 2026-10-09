"""
AeroScript 2.0 — Toolbar
Renders the Apple-minimal top toolbar and right-side tool panel.
All rendering is done in-place on the OpenCV frame.
"""

import cv2
import numpy as np
import ui_icons
from ui import themes
from ui.themes import draw_rounded_rect
from config import settings


# Tool icons and labels
TOOL_ICON_NAMES = {
    settings.TOOL_PEN:         "pen",
    settings.TOOL_PENCIL:      "pen",
    settings.TOOL_MARKER:      "marker",
    settings.TOOL_HIGHLIGHTER: "highlighter",
    settings.TOOL_ERASER:      "eraser",
    settings.TOOL_SHAPE_PEN:   "shape_pen",
}

# Color palette displayed in toolbar (name, BGR)
COLOR_PALETTE = [
    ("R", (0,   0,   220)),
    ("G", (0,   200,  50)),
    ("B", (220,  30,   0)),
    ("Y", (0,   220, 255)),
    ("O", (0,   140, 255)),
    ("W", (240, 240, 240)),
]


class Toolbar:
    """
    Renders the full AeroScript UI overlay on a BGR frame.

    Layout:
        Top bar (full width, 80px tall):
            [AeroScript logo] [color swatches] [eraser] [whiteboard toggle] [mode indicator]
        Right panel (80px wide, below toolbar):
            [tool buttons: Pen, Pencil, Marker, Highlighter, Shape, Eraser]
        Bottom-center (when pinch active):
            [BIN zone - red]

    All coordinates are dynamically computed from frame dimensions.
    """

    def __init__(self):
        self._last_frame_w = 0
        self._last_frame_h = 0
        # Precomputed layout rects, updated when frame size changes
        self._layout = {}

    # ── Public render entry point ─────────────────────────────────────────────

    def render(self, frame: np.ndarray, state) -> dict:
        """
        Draw the complete toolbar UI onto the frame (in-place).

        Args:
            frame : BGR frame to render onto
            state : AppState instance

        Returns:
            layout dict with zone rects for hit-testing (used by input handler)
        """
        h, w = frame.shape[:2]
        if w != self._last_frame_w or h != self._last_frame_h:
            self._last_frame_w = w
            self._last_frame_h = h
            self._layout = self._compute_layout(w, h)

        self._draw_top_bar(frame, state, w, h)
        self._draw_side_panel(frame, state, w, h)
        return self._layout

    def get_layout(self) -> dict:
        return self._layout

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _draw_top_bar(self, frame: np.ndarray, state, w: int, h: int):
        """Draw the dark top toolbar with colors and controls."""
        bar_h = themes.TOOLBAR_H

        # Dark frosted-glass panel
        panel = np.full((bar_h, w, 3), themes.BG_DARK, dtype=np.uint8)
        themes.overlay_alpha(frame, panel, 0, 0, alpha=0.88)

        # Bottom divider line
        cv2.line(frame, (0, bar_h), (w, bar_h), themes.SEPARATOR, 1)

        # ── Logo / Brand text ──────────────────────────────────────────────
        cv2.putText(frame, "AeroScript", (16, 50),
                    themes.FONT_BOLD, themes.FONT_LARGE,
                    themes.ACCENT_CYAN, themes.FONT_THICK_NORMAL, cv2.LINE_AA)

        # ── Color swatches ─────────────────────────────────────────────────
        swatch_start_x = 190
        sw = themes.SWATCH_SIZE
        sm = themes.SWATCH_MARGIN

        for i, (name, color) in enumerate(COLOR_PALETTE):
            cx = swatch_start_x + i * (sw + sm) + sw // 2
            cy = bar_h // 2

            # Highlight active color
            if color == state.current_color:
                cv2.circle(frame, (cx, cy), sw // 2 + 4, themes.ACCENT_CYAN, 2)

            cv2.circle(frame, (cx, cy), sw // 2, color, -1)

        # ── Eraser button ──────────────────────────────────────────────────
        eraser_x = swatch_start_x + len(COLOR_PALETTE) * (sw + sm) + 16
        ex1, ey1 = eraser_x, 18
        ex2, ey2 = eraser_x + 50, bar_h - 18

        if state.eraser_mode:
            draw_rounded_rect(frame, ex1, ey1, ex2, ey2, themes.ACCENT_CYAN, radius=8)
            text_col = themes.BG_DARK
        else:
            draw_rounded_rect(frame, ex1, ey1, ex2, ey2, themes.BG_PANEL_LIGHT, radius=8)
            text_col = themes.TEXT_PRIMARY

        ui_icons.draw_icon(frame, "eraser", (ex1 + ex2) // 2, (ey1 + ey2) // 2, 20, text_col, 2)

        # ── Whiteboard toggle ──────────────────────────────────────────────
        wb_x1 = ex2 + 12
        wb_x2 = wb_x1 + 50
        wb_y1, wb_y2 = 18, bar_h - 18

        if state.white_mode:
            draw_rounded_rect(frame, wb_x1, wb_y1, wb_x2, wb_y2, themes.TEXT_PRIMARY, radius=8)
            wbtxt_col = themes.BG_DARK
        else:
            draw_rounded_rect(frame, wb_x1, wb_y1, wb_x2, wb_y2, themes.BG_PANEL_LIGHT, radius=8)
            wbtxt_col = themes.TEXT_PRIMARY

        ui_icons.draw_icon(frame, "whiteboard", (wb_x1 + wb_x2) // 2, (wb_y1 + wb_y2) // 2, 20, wbtxt_col, 2)

        # ── Brush size indicator ───────────────────────────────────────────
        bs_x = wb_x2 + 20
        bs_label = f"B:{state.brush_size:2d}"
        cv2.putText(frame, bs_label, (bs_x, bar_h // 2 + 6),
                    themes.FONT, themes.FONT_SMALL, themes.TEXT_SECONDARY,
                    themes.FONT_THICK_THIN, cv2.LINE_AA)

        # ── Mode pill (top right) ──────────────────────────────────────────
        mode_text = state.app_mode.upper()
        tx, _ = cv2.getTextSize(mode_text, themes.FONT, themes.FONT_SMALL, 1)
        mode_x1 = w - tx[0] - 28
        mode_x2 = w - 8
        draw_rounded_rect(frame, mode_x1, 22, mode_x2, bar_h - 22,
                          themes.ACCENT_GREEN, radius=8)
        cv2.putText(frame, mode_text, (mode_x1 + 6, bar_h - 28),
                    themes.FONT, themes.FONT_SMALL, themes.BG_DARK,
                    themes.FONT_THICK_THIN, cv2.LINE_AA)

        # Store zone rects for hit-testing
        self._layout["colors"] = [
            (swatch_start_x + i * (sw + sm), 0,
             swatch_start_x + i * (sw + sm) + sw, bar_h, color)
            for i, (_, color) in enumerate(COLOR_PALETTE)
        ]
        self._layout["eraser"] = (ex1, ey1, ex2, ey2)
        self._layout["whiteboard"] = (wb_x1, wb_y1, wb_x2, wb_y2)
        self._layout["bar_h"]   = bar_h

    # ── Side panel ────────────────────────────────────────────────────────────

    def _draw_side_panel(self, frame: np.ndarray, state, w: int, h: int):
        """Draw the right-side tool selection panel."""
        bar_h  = themes.TOOLBAR_H
        sw     = themes.SIDE_W
        panel_y1 = bar_h + 1
        panel_h  = h - bar_h

        # Panel background
        panel = np.full((panel_h, sw, 3), themes.BG_PANEL, dtype=np.uint8)
        themes.overlay_alpha(frame, panel, w - sw, panel_y1, alpha=0.85)
        cv2.line(frame, (w - sw, panel_y1), (w - sw, h), themes.SEPARATOR, 1)

        tools_in_order = list(TOOL_ICON_NAMES.items())
        btn_h = 52
        padding = 8
        active_tool = state.effective_tool()

        self._layout["tools"] = []

        for i, (tool_name, icon_key) in enumerate(tools_in_order):
            ty1 = panel_y1 + i * (btn_h + padding) + padding
            ty2 = ty1 + btn_h
            tx1 = w - sw + 6
            tx2 = w - 6

            if ty2 > h - 8:
                break  # don't overflow

            is_active = (tool_name == active_tool)
            if is_active:
                draw_rounded_rect(frame, tx1, ty1, tx2, ty2, themes.ACCENT_CYAN, radius=8)
                text_col = themes.BG_DARK
            else:
                draw_rounded_rect(frame, tx1, ty1, tx2, ty2, themes.BG_PANEL_LIGHT, radius=8)
                text_col = themes.TEXT_PRIMARY

            ui_icons.draw_icon(frame, icon_key, (tx1 + tx2) // 2, (ty1 + ty2) // 2, 22, text_col, 2)
            self._layout["tools"].append((tx1, ty1, tx2, ty2, tool_name))

    # ── Trash bin (shown on pinch) ────────────────────────────────────────────

    def draw_bin(self, frame: np.ndarray, cx: int, cy: int):
        """
        Draw the trash bin zone at the bottom center when pinch is active.
        Migrated from original test.py bin logic.
        """
        h, w = frame.shape[:2]
        bin_w, bin_h = 110, 60
        bx1 = w // 2 - bin_w // 2
        by1 = h - bin_h
        bx2 = bx1 + bin_w
        by2 = h

        in_bin = bx1 < cx < bx2 and by1 < cy < by2
        fill_color = themes.ACCENT_RED if in_bin else (70, 70, 200)

        draw_rounded_rect(frame, bx1, by1, bx2, by2, fill_color, radius=10)
        cv2.putText(frame, "BIN", (bx1 + 30, by1 + 38),
                    themes.FONT, themes.FONT_MEDIUM, themes.TEXT_PRIMARY,
                    themes.FONT_THICK_NORMAL, cv2.LINE_AA)

        return (bx1, by1, bx2, by2)

    # ── Layout ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_layout(w: int, h: int) -> dict:
        """Pre-compute static layout zones from frame dimensions."""
        bar_h = themes.TOOLBAR_H
        return {
            "bar_h": bar_h,
            "side_w": themes.SIDE_W,
            "canvas_rect": (0, bar_h, w - themes.SIDE_W, h),
        }

    # ── Hit Testing ───────────────────────────────────────────────────────────

    @staticmethod
    def hit_test_color(lx: dict, layout: dict) -> tuple | None:
        """
        Check if finger position hits a color swatch. Returns color BGR or None.
        """
        from vision.hand_tracker import HandTracker as LM
        if not lx:
            return None
        cx, cy = lx[LM.INDEX_TIP]
        bar_h = layout.get("bar_h", themes.TOOLBAR_H)
        if cy > bar_h:
            return None
        for (x1, _, x2, __, color) in layout.get("colors", []):
            if x1 < cx < x2:
                return color
        return None

    @staticmethod
    def hit_test_zone(cx: int, cy: int, rect: tuple) -> bool:
        """Return True if (cx, cy) falls inside rect (x1,y1,x2,y2)."""
        x1, y1, x2, y2 = rect
        return x1 < cx < x2 and y1 < cy < y2

    @staticmethod
    def hit_test_tool(cx: int, cy: int, layout: dict) -> str | None:
        """Return tool name if (cx, cy) hits a tool button, else None."""
        for (tx1, ty1, tx2, ty2, tool_name) in layout.get("tools", []):
            if tx1 < cx < tx2 and ty1 < cy < ty2:
                return tool_name
        return None
