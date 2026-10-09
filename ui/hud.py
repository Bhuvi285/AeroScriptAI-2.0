"""
AeroScript 2.0 — HUD (Heads-Up Display)
Renders real-time performance and status overlays.
Minimal, non-intrusive, Apple-inspired information display.
"""

import cv2
import numpy as np
from ui import themes
from ui.themes import draw_rounded_rect


class HUD:
    """
    Renders the minimal HUD overlay on the frame.

    Displays:
    - FPS counter (top-left below logo)
    - Current gesture + confidence (bottom-left)
    - Current tool + color dot (bottom-left)
    - Transient status messages (center-top, fades)
    - Voice command status (center-bottom)
    - Undo/Redo depth indicator
    - Performance metrics (when debug mode on)
    """

    def render(self, frame: np.ndarray, state, perf_stats: dict,
               canvas_stats: dict):
        """
        Draw all HUD elements onto the frame.

        Args:
            frame       : BGR frame (in-place)
            state       : AppState
            perf_stats  : dict from PerformanceMonitor (fps, latency_ms, etc.)
            canvas_stats: dict with undo_depth, redo_depth, pixel_coverage
        """
        h, w = frame.shape[:2]
        bar_h = themes.TOOLBAR_H
        side_w = themes.SIDE_W

        # ── FPS badge (top-left, inside toolbar) ──────────────────────────
        fps = perf_stats.get("fps", 0)
        fps_color = themes.ACCENT_GREEN if fps >= 25 else themes.ACCENT_ORANGE
        cv2.putText(frame, f"{fps:.0f} FPS",
                    (16, bar_h - 14),
                    themes.FONT, themes.FONT_SMALL,
                    fps_color, themes.FONT_THICK_THIN, cv2.LINE_AA)

        # ── Gesture + confidence badge (bottom-left) ──────────────────────
        gesture_text = state.current_gesture.upper().replace("_", " ")
        conf_pct     = int(state.gesture_confidence * 100)

        gesture_y = h - themes.STATUS_BAR_H - 12
        self._draw_badge(frame,
                         x=16, y=gesture_y,
                         label="GESTURE",
                         value=f"{gesture_text}  {conf_pct}%",
                         label_color=themes.TEXT_SECONDARY,
                         value_color=themes.ACCENT_CYAN)

        # ── Tool indicator (bottom-left, above gesture) ───────────────────
        tool_y = gesture_y - 26
        tool_label = state.effective_tool().upper()
        self._draw_badge(frame,
                         x=16, y=tool_y,
                         label="TOOL",
                         value=tool_label,
                         label_color=themes.TEXT_SECONDARY,
                         value_color=themes.TEXT_PRIMARY)

        # ── Color dot ─────────────────────────────────────────────────────
        col_cx = 16 + 46 + 14    # after "TOOL: MARKER" text
        cv2.circle(frame, (col_cx + 70, tool_y - 4), 7,
                   state.current_color, -1)
        cv2.circle(frame, (col_cx + 70, tool_y - 4), 7,
                   themes.TEXT_SECONDARY, 1)

        # ── Undo depth pill ───────────────────────────────────────────────
        if canvas_stats.get("undo_depth", 0) > 0:
            ud = canvas_stats["undo_depth"]
            rd = canvas_stats.get("redo_depth", 0)
            self._draw_badge(frame,
                             x=16, y=tool_y - 26,
                             label="HIST",
                             value=f"↩{ud}  ↪{rd}",
                             label_color=themes.TEXT_SECONDARY,
                             value_color=themes.TEXT_SECONDARY)

        # ── Transient status message (center, below toolbar) ──────────────
        status = state.get_active_status()
        if status:
            self._draw_status_banner(frame, status, w, bar_h)

        # ── Voice command display ──────────────────────────────────────────
        vcmd = state.get_voice_status()
        if vcmd:
            self._draw_voice_banner(frame, vcmd, w, h)

        # ── Accessibility indicator ────────────────────────────────────────
        if state.accessibility_mode:
            cv2.putText(frame, "A11Y", (w - side_w - 60, bar_h + 20),
                        themes.FONT, themes.FONT_SMALL,
                        themes.ACCENT_ORANGE, themes.FONT_THICK_THIN, cv2.LINE_AA)

        # ── Debug overlay ─────────────────────────────────────────────────
        if state.show_debug:
            self._draw_debug(frame, perf_stats, canvas_stats, w, h, bar_h, side_w)

        # ── Bottom status bar ─────────────────────────────────────────────
        self._draw_status_bar(frame, state, w, h, side_w)

    # ── Sub-renderers ─────────────────────────────────────────────────────────

    @staticmethod
    def _draw_badge(frame, x, y, label, value,
                    label_color=themes.TEXT_SECONDARY,
                    value_color=themes.TEXT_PRIMARY):
        """Render a small label: value pair."""
        cv2.putText(frame, f"{label}:", (x, y),
                    themes.FONT, themes.FONT_SMALL,
                    label_color, themes.FONT_THICK_THIN, cv2.LINE_AA)
        tw, _ = cv2.getTextSize(f"{label}:", themes.FONT, themes.FONT_SMALL, 1)
        cv2.putText(frame, value, (x + tw[0] + 6, y),
                    themes.FONT_BOLD, themes.FONT_SMALL,
                    value_color, themes.FONT_THICK_THIN, cv2.LINE_AA)

    @staticmethod
    def _draw_status_banner(frame, text, frame_w, bar_h):
        """Centered transient status toast below toolbar."""
        tw, th = cv2.getTextSize(text, themes.FONT_BOLD, themes.FONT_MEDIUM, 2)[:2]
        bx1 = (frame_w - tw[0]) // 2 - 16
        by1 = bar_h + 12
        bx2 = bx1 + tw[0] + 32
        by2 = by1 + th + 20

        draw_rounded_rect(frame, bx1, by1, bx2, by2, themes.BG_DARK, radius=10)
        cv2.putText(frame, text,
                    (bx1 + 16, by2 - 10),
                    themes.FONT_BOLD, themes.FONT_MEDIUM,
                    themes.ACCENT_CYAN, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_voice_banner(frame, command, frame_w, frame_h):
        """Centered voice command toast near the bottom."""
        display = f"Voice: {command}"
        tw, th = cv2.getTextSize(display, themes.FONT_BOLD, themes.FONT_SMALL, 1)[:2]
        bx1 = (frame_w - tw[0]) // 2 - 16
        by1 = frame_h - themes.STATUS_BAR_H - th - 24
        bx2 = bx1 + tw[0] + 32
        by2 = by1 + th + 20

        draw_rounded_rect(frame, bx1, by1, bx2, by2, themes.BG_DARK, radius=8)
        cv2.putText(frame, display,
                    (bx1 + 16, by2 - 10),
                    themes.FONT_BOLD, themes.FONT_SMALL,
                    themes.ACCENT_GREEN, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_status_bar(frame, state, w, h, side_w):
        """Thin bottom status bar with keyboard shortcuts hint."""
        sb_y = h - themes.STATUS_BAR_H
        bar  = np.full((themes.STATUS_BAR_H, w, 3), themes.BG_DARK, dtype=np.uint8)
        themes.overlay_alpha(frame, bar, 0, sb_y, alpha=0.75)

        hints = "C:Clear  Ctrl+Z:Undo  Ctrl+Y:Redo  +:Brush+  -:Brush-  ESC:Exit"
        cv2.putText(frame, hints, (16, h - 8),
                    themes.FONT, 0.38,
                    themes.TEXT_DISABLED, 1, cv2.LINE_AA)

        # HWR mode indicator
        if state.handwriting_mode:
            cv2.putText(frame, "HWR MODE", (w - side_w - 100, h - 8),
                        themes.FONT, themes.FONT_SMALL,
                        themes.ACCENT_ORANGE, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_debug(frame, perf_stats, canvas_stats, w, h, bar_h, side_w):
        """Full debug overlay (enabled with D key)."""
        lines = [
            f"FPS:       {perf_stats.get('fps', 0):.1f}",
            f"Latency:   {perf_stats.get('latency_ms', 0):.1f} ms",
            f"CPU:       {perf_stats.get('cpu_pct', 0):.1f}%",
            f"MEM:       {perf_stats.get('mem_mb', 0):.0f} MB",
            f"UndoStack: {canvas_stats.get('undo_depth', 0)}",
            f"Coverage:  {canvas_stats.get('pixel_coverage', 0)*100:.1f}%",
        ]
        dx, dy = w - side_w - 180, bar_h + 12
        panel = np.full((len(lines) * 20 + 12, 175, 3), themes.BG_DARK, dtype=np.uint8)
        themes.overlay_alpha(frame, panel, dx - 4, dy - 4, alpha=0.8)

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (dx, dy + i * 20),
                        themes.FONT_MONO, themes.FONT_SMALL,
                        themes.TEXT_SECONDARY, themes.FONT_THICK_THIN, cv2.LINE_AA)
