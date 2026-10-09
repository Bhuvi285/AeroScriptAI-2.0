"""
AeroScript 2.0 — Drawing Engine
Stroke interpolation, brush styles, and drawing primitives.
Migrates and extends the drawing logic from the original test.py.
"""

import cv2
import numpy as np
from config import settings


class DrawingEngine:
    """
    Handles all rendering operations on a canvas surface.

    Supported tools:
    - Pen        : solid smooth stroke (original behavior)
    - Pencil     : slightly textured, thinner stroke
    - Marker     : thick, semi-transparent stroke
    - Highlighter: very wide, low-opacity color band
    - Eraser     : circular erase region

    All methods draw directly onto the provided numpy array in-place.
    """

    # ── Tool Style Definitions ────────────────────────────────────────────────

    TOOL_STYLES = {
        settings.TOOL_PEN: {
            "size_multiplier": 1.0,
            "alpha": 1.0,
            "jitter": 0,
        },
        settings.TOOL_PENCIL: {
            "size_multiplier": 0.6,
            "alpha": 0.85,
            "jitter": 1,       # slight randomness to simulate pencil grain
        },
        settings.TOOL_MARKER: {
            "size_multiplier": 2.5,
            "alpha": 0.92,
            "jitter": 0,
        },
        settings.TOOL_HIGHLIGHTER: {
            "size_multiplier": 5.0,
            "alpha": 0.35,
            "jitter": 0,
        },
        settings.TOOL_ERASER: {
            "size_multiplier": 1.0,
            "alpha": 1.0,
            "jitter": 0,
        },
    }

    # ── Core Drawing ──────────────────────────────────────────────────────────

    @staticmethod
    def draw_stroke(surface: np.ndarray,
                    x1: int, y1: int,
                    x2: int, y2: int,
                    color: tuple,
                    brush_size: int,
                    tool: str = settings.TOOL_PEN):
        """
        Draw a smooth stroke from (x1,y1) to (x2,y2) using Euclidean interpolation.
        This is the core drawing primitive, migrated from test.py.

        Args:
            surface   : canvas numpy array (modified in-place)
            x1, y1   : start point
            x2, y2   : end point
            color     : BGR tuple
            brush_size: brush radius in pixels
            tool      : one of TOOL_* constants
        """
        style   = DrawingEngine.TOOL_STYLES.get(tool, DrawingEngine.TOOL_STYLES[settings.TOOL_PEN])
        radius  = max(1, int(brush_size * style["size_multiplier"]))
        alpha   = style["alpha"]
        jitter  = style["jitter"]

        dx   = x2 - x1
        dy   = y2 - y1
        dist = max(1, int(np.hypot(dx, dy)))

        if alpha >= 1.0:
            # Fast path: direct draw (original test.py behavior)
            for i in range(dist):
                x = int(x1 + dx * i / dist)
                y = int(y1 + dy * i / dist)
                if jitter:
                    x += int(np.random.uniform(-jitter, jitter))
                    y += int(np.random.uniform(-jitter, jitter))
                cv2.circle(surface, (x, y), radius, color, -1)
        else:
            # Slow path: alpha-blended overlay for marker/highlighter
            overlay = surface.copy()
            for i in range(dist):
                x = int(x1 + dx * i / dist)
                y = int(y1 + dy * i / dist)
                cv2.circle(overlay, (x, y), radius, color, -1)
            cv2.addWeighted(overlay, alpha, surface, 1 - alpha, 0, surface)

    @staticmethod
    def erase(surface: np.ndarray, x: int, y: int,
              eraser_size: int = settings.ERASER_SIZE):
        """
        Erase a circular region at (x, y).
        Migrated from original test.py — same 30px default radius.
        """
        cv2.circle(surface, (x, y), eraser_size, (0, 0, 0), -1)

    @staticmethod
    def erase_stroke(surface: np.ndarray,
                     x1: int, y1: int,
                     x2: int, y2: int,
                     eraser_size: int = settings.ERASER_SIZE):
        """Erase along a stroke path (for smooth erasing while moving)."""
        dx   = x2 - x1
        dy   = y2 - y1
        dist = max(1, int(np.hypot(dx, dy)))
        for i in range(dist):
            x = int(x1 + dx * i / dist)
            y = int(y1 + dy * i / dist)
            cv2.circle(surface, (x, y), eraser_size, (0, 0, 0), -1)

    # ── Shape Drawing (for auto-correct) ─────────────────────────────────────

    @staticmethod
    def draw_perfect_circle(surface: np.ndarray, cx: int, cy: int,
                            radius: int, color: tuple, thickness: int = 3):
        cv2.circle(surface, (cx, cy), radius, color, thickness)

    @staticmethod
    def draw_perfect_rect(surface: np.ndarray,
                          x: int, y: int, w: int, h: int,
                          color: tuple, thickness: int = 3):
        cv2.rectangle(surface, (x, y), (x + w, y + h), color, thickness)

    @staticmethod
    def draw_perfect_polygon(surface: np.ndarray,
                             pts: np.ndarray,
                             color: tuple, thickness: int = 3):
        cv2.drawContours(surface, [pts], -1, color, thickness)

    @staticmethod
    def draw_arrow(surface: np.ndarray,
                   pt1: tuple, pt2: tuple,
                   color: tuple, thickness: int = 3):
        cv2.arrowedLine(surface, pt1, pt2, color, thickness, tipLength=0.3)

    @staticmethod
    def draw_line(surface: np.ndarray,
                  pt1: tuple, pt2: tuple,
                  color: tuple, thickness: int = 3):
        cv2.line(surface, pt1, pt2, color, thickness)

    # ── Cursor ────────────────────────────────────────────────────────────────

    @staticmethod
    def draw_cursor(frame: np.ndarray, x: int, y: int,
                    color: tuple = (0, 255, 255), large: bool = False):
        """
        Draw the fingertip cursor (black outer ring + colored inner dot).
        Migrated from test.py. large=True for accessibility mode.
        """
        outer = 14 if large else 10
        inner = 9  if large else 6
        cv2.circle(frame, (x, y), outer, (0, 0, 0), -1)
        cv2.circle(frame, (x, y), inner, color, -1)

    # ── Erase Rough Stroke Points ─────────────────────────────────────────────

    @staticmethod
    def erase_points(surface: np.ndarray, points: list, brush_size: int):
        """
        Erase all points and segments in a stroke (used to delete rough stroke before shape autocorrect).
        Migrated and enhanced from original test.py shape-replacement logic.
        """
        if not points:
            return
        radius = max(brush_size + 4, 8)
        for i in range(len(points)):
            x2, y2 = points[i]
            if i > 0:
                x1, y1 = points[i - 1]
                cv2.line(surface, (x1, y1), (x2, y2), (0, 0, 0), radius * 2)
            cv2.circle(surface, (x2, y2), radius, (0, 0, 0), -1)

