"""
AeroScript 2.0 — Startup Screen
Mode selector rendered entirely in OpenCV — no external GUI framework.
Apple-inspired centered card layout with minimal text and clean spacing.
"""

import cv2
import numpy as np
from ui import themes
from ui.themes import draw_rounded_rect
from config import settings


class StartupScreen:
    """
    Renders the AeroScript startup mode-selection screen.

    Modes:
        0 → Smart Canvas
        1 → Presentation Mode
        2 → Screen Annotation
        3 → Settings
        4 → Exit

    Navigation:
        - Keyboard: 1-5 to select, Enter to confirm, Up/Down arrows
        - Returns the selected mode constant on confirm, "exit" to quit.
    """

    MENU_ITEMS = [
        (settings.MODE_CANVAS,        "Smart Canvas",        "Draw, sketch and annotate"),
        (settings.MODE_PRESENTATION,  "Presentation Mode",   "Control slides with gestures"),
        (settings.MODE_ANNOTATION,    "Screen Annotation",   "Draw over your desktop"),
        (settings.MODE_TESTING,       "Evaluation Mode",     "Measure accuracy & latency"),
        ("exit",                      "Exit",                "Quit AeroScript"),
    ]

    CANVAS_W = 1280
    CANVAS_H = 720

    def __init__(self):
        self._selected_idx = 0

    def run(self) -> str:
        """
        Display the startup screen and wait for user input.

        Returns:
            mode string (e.g. settings.MODE_CANVAS) or "exit"
        """
        cv2.namedWindow("AeroScript", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AeroScript", self.CANVAS_W, self.CANVAS_H)

        while True:
            frame = self._render(self.CANVAS_W, self.CANVAS_H)
            cv2.imshow("AeroScript", frame)

            key = cv2.waitKey(30) & 0xFF

            if key == 27:   # ESC
                return "exit"
            elif key == 13 or key == 10:   # Enter
                mode, _, _ = self.MENU_ITEMS[self._selected_idx]
                return mode
            elif key == 82 or key == ord('w'):  # Up arrow / w
                self._selected_idx = (self._selected_idx - 1) % len(self.MENU_ITEMS)
            elif key == 84 or key == ord('s'):  # Down arrow / s
                self._selected_idx = (self._selected_idx + 1) % len(self.MENU_ITEMS)
            elif ord('1') <= key <= ord('5'):
                idx = key - ord('1')
                if idx < len(self.MENU_ITEMS):
                    self._selected_idx = idx
            elif key == ord('q'):
                return "exit"

            # Check window close button
            if cv2.getWindowProperty("AeroScript", cv2.WND_PROP_VISIBLE) < 1:
                return "exit"

    def _render(self, w: int, h: int) -> np.ndarray:
        """Render the startup screen to a BGR frame."""
        frame = np.full((h, w, 3), themes.BG_DARK, dtype=np.uint8)

        # ── Animated gradient background (simple scanline) ─────────────────
        for y in range(h):
            alpha = y / h
            r = int(20 + 15 * alpha)
            g = int(20 + 8  * alpha)
            b = int(25 + 10 * alpha)
            frame[y, :] = (r, g, b)

        # ── Central card ──────────────────────────────────────────────────
        card_w, card_h = 520, 460
        cx1 = (w - card_w) // 2
        cy1 = (h - card_h) // 2
        cx2 = cx1 + card_w
        cy2 = cy1 + card_h

        card = np.full((card_h, card_w, 3), themes.BG_PANEL, dtype=np.uint8)
        themes.overlay_alpha(frame, card, cx1, cy1, alpha=0.92)
        draw_rounded_rect(frame, cx1, cy1, cx2, cy2, themes.SEPARATOR,
                          radius=18, filled=False, thickness=1)

        # ── Accent line at top of card ─────────────────────────────────────
        cv2.line(frame, (cx1 + 20, cy1), (cx2 - 20, cy1), themes.ACCENT_CYAN, 2)

        # ── Logo text ──────────────────────────────────────────────────────
        logo_y = cy1 + 60
        cv2.putText(frame, "AeroScript",
                    (cx1 + 30, logo_y),
                    themes.FONT_BOLD, themes.FONT_XLARGE,
                    themes.ACCENT_CYAN, themes.FONT_THICK_NORMAL, cv2.LINE_AA)

        subtitle = "AI-Powered Touchless Interaction"
        cv2.putText(frame, subtitle,
                    (cx1 + 30, logo_y + 32),
                    themes.FONT, themes.FONT_SMALL,
                    themes.TEXT_SECONDARY, themes.FONT_THICK_THIN, cv2.LINE_AA)

        # ── Divider ────────────────────────────────────────────────────────
        div_y = logo_y + 50
        cv2.line(frame, (cx1 + 20, div_y), (cx2 - 20, div_y), themes.SEPARATOR, 1)

        # ── Menu items ─────────────────────────────────────────────────────
        item_start_y = div_y + 24
        item_h       = 58

        for i, (mode, title, subtitle_txt) in enumerate(self.MENU_ITEMS):
            iy1 = item_start_y + i * item_h
            iy2 = iy1 + item_h - 6
            ix1 = cx1 + 18
            ix2 = cx2 - 18

            is_selected = (i == self._selected_idx)

            if is_selected:
                draw_rounded_rect(frame, ix1, iy1, ix2, iy2,
                                  themes.BG_SELECTED, radius=10)
                cv2.line(frame, (ix1, iy1 + 4), (ix1, iy2 - 4),
                         themes.ACCENT_CYAN, 3)

            # Number badge
            badge_col = themes.ACCENT_CYAN if is_selected else themes.TEXT_DISABLED
            cv2.putText(frame, str(i + 1),
                        (ix1 + 12, iy1 + 32),
                        themes.FONT_BOLD, themes.FONT_MEDIUM,
                        badge_col, themes.FONT_THICK_THIN, cv2.LINE_AA)

            # Title
            title_col = themes.TEXT_PRIMARY if is_selected else themes.TEXT_SECONDARY
            cv2.putText(frame, title,
                        (ix1 + 40, iy1 + 26),
                        themes.FONT_BOLD, themes.FONT_MEDIUM,
                        title_col, themes.FONT_THICK_THIN, cv2.LINE_AA)

            # Subtitle
            cv2.putText(frame, subtitle_txt,
                        (ix1 + 40, iy1 + 44),
                        themes.FONT, 0.38,
                        themes.TEXT_DISABLED, 1, cv2.LINE_AA)

        # ── Footer hint ────────────────────────────────────────────────────
        hint = "ENTER to select   W/S or arrows to navigate   ESC to exit"
        hw, _ = cv2.getTextSize(hint, themes.FONT, 0.38, 1)[:2]
        cv2.putText(frame, hint,
                    ((w - hw[0]) // 2, h - 20),
                    themes.FONT, 0.38,
                    themes.TEXT_DISABLED, 1, cv2.LINE_AA)

        return frame
