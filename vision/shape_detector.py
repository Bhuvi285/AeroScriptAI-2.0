"""
AeroScript 2.0 — Shape Detector
Extended shape recognition supporting 8 shape types.
Migrates and extends the original detect_shape() from test.py.

Shapes supported:
    Line, Circle, Triangle, Square, Rectangle, Pentagon, Hexagon, Arrow

Algorithm:
    1. Check for Line first (before convex hull distorts it)
    2. Convex hull the stroke points (smooths gaps and noise)
    3. Douglas-Peucker polygon approximation
    4. Classify by vertex count + geometric properties (circularity, aspect ratio, radial variance)
"""

import cv2
import numpy as np
import logging
from config import settings

logger = logging.getLogger(__name__)

# Shape name constants
SHAPE_UNKNOWN   = "Unknown"
SHAPE_LINE      = "Line"
SHAPE_CIRCLE    = "Circle"
SHAPE_TRIANGLE  = "Triangle"
SHAPE_SQUARE    = "Square"
SHAPE_RECTANGLE = "Rectangle"
SHAPE_PENTAGON  = "Pentagon"
SHAPE_HEXAGON   = "Hexagon"
SHAPE_ARROW     = "Arrow"


def detect_shape(points: list) -> tuple[str, np.ndarray | None]:
    """
    Detect the geometric shape represented by a list of (x,y) stroke points.

    Args:
        points: list of (x, y) tuples — the raw stroke trajectory

    Returns:
        (shape_name: str, approx_contour: np.ndarray | None)
        shape_name will be SHAPE_UNKNOWN if no shape is detected.
    """
    min_pts = getattr(settings, "SHAPE_MIN_POINTS", 25)
    if not points or len(points) < min_pts:
        return SHAPE_UNKNOWN, None

    pts_array = np.array(points, dtype=np.int32).reshape((-1, 2))
    if len(pts_array) < 5:
        return SHAPE_UNKNOWN, None

    # ── Check for Line first (before convex hull distorts it) ─────────────────
    rect = cv2.minAreaRect(pts_array)
    _, (rw, rh), _ = rect
    long_side  = max(rw, rh)
    short_side = min(rw, rh) + 1e-6
    if long_side / short_side > 3.8 and short_side < 45:
        return SHAPE_LINE, None

    # ── Check raw points for arrow shape (7-vertex block arrow) ───────────────
    raw_contour = pts_array.reshape((-1, 1, 2))
    raw_peri = cv2.arcLength(raw_contour, True)
    if raw_peri > 10:
        epsilon_raw = getattr(settings, "SHAPE_APPROX_EPSILON", 0.03) * raw_peri
        approx_raw = cv2.approxPolyDP(raw_contour, epsilon_raw, True)
        if len(approx_raw) == 7 and _is_arrow(approx_raw):
            return SHAPE_ARROW, approx_raw

    # ── Convex hull ───────────────────────────────────────────────────────────
    hull = cv2.convexHull(raw_contour)
    peri = cv2.arcLength(hull, True)
    if peri < 10:
        return SHAPE_UNKNOWN, None

    epsilon = getattr(settings, "SHAPE_APPROX_EPSILON", 0.03) * peri
    approx  = cv2.approxPolyDP(hull, epsilon, True)
    sides   = len(approx)

    area = cv2.contourArea(hull)
    circularity = (4.0 * np.pi * area) / (peri * peri + 1e-6)
    circ_min = getattr(settings, "CIRCLE_CIRCULARITY_MIN", 0.60)

    # Radial variance from minimum enclosing circle (smooth circles have very low variance)
    (cx, cy), radius = cv2.minEnclosingCircle(hull)
    if radius > 0:
        dists = np.hypot(hull[:, 0, 0] - cx, hull[:, 0, 1] - cy)
        radial_cv = float(np.std(dists) / (np.mean(dists) + 1e-6))
    else:
        radial_cv = 1.0

    # Bounding rect aspect ratio
    bx, by, bw, bh = cv2.boundingRect(hull)
    ar = bw / float(bh + 1e-6)

    # ── Classification ────────────────────────────────────────────────────────

    # 1. Circle check (prioritized for smooth hand-drawn circles with 5+ vertices)
    if sides >= 5 and 0.70 <= ar <= 1.40 and circularity >= circ_min and radial_cv < 0.12:
        return SHAPE_CIRCLE, approx

    if sides == 3:
        return SHAPE_TRIANGLE, approx

    elif sides == 4:
        if 0.80 <= ar <= 1.25:
            return SHAPE_SQUARE, approx
        else:
            return SHAPE_RECTANGLE, approx

    elif sides == 5:
        if circularity >= 0.85 and 0.75 <= ar <= 1.33:
            return SHAPE_CIRCLE, approx
        return SHAPE_PENTAGON, approx

    elif sides == 6:
        if circularity >= 0.85 and 0.75 <= ar <= 1.33:
            return SHAPE_CIRCLE, approx
        return SHAPE_HEXAGON, approx

    elif sides == 7:
        if _is_arrow(approx):
            return SHAPE_ARROW, approx
        if circularity >= circ_min:
            return SHAPE_CIRCLE, approx
        return SHAPE_UNKNOWN, approx

    elif sides > 7:
        if circularity >= circ_min:
            return SHAPE_CIRCLE, approx

    return SHAPE_UNKNOWN, approx


def draw_perfect_shape(canvas: np.ndarray, shape: str, approx: np.ndarray | None,
                       contour: np.ndarray | None, color: tuple, thickness: int = 3):
    """
    Replace a rough stroke with a clean, perfect shape on the canvas.

    Args:
        canvas   : drawing surface (modified in-place)
        shape    : one of the SHAPE_* constants
        approx   : approximated contour from detect_shape()
        contour  : the raw point contour
        color    : BGR color tuple
        thickness: line thickness
    """
    if canvas is None or shape == SHAPE_UNKNOWN:
        return

    pts = contour if contour is not None else approx

    if shape == SHAPE_CIRCLE:
        if pts is not None and len(pts) > 0:
            (x, y), r = cv2.minEnclosingCircle(pts)
            cv2.circle(canvas, (int(round(x)), int(round(y))), max(1, int(round(r))), color, thickness)

    elif shape == SHAPE_SQUARE:
        if pts is not None and len(pts) > 0:
            x, y, w, h = cv2.boundingRect(pts)
            side = int(round((w + h) / 2))
            cx, cy = x + w // 2, y + h // 2
            half = side // 2
            cv2.rectangle(canvas, (cx - half, cy - half), (cx - half + side, cy - half + side), color, thickness)

    elif shape == SHAPE_RECTANGLE:
        if pts is not None and len(pts) > 0:
            x, y, w, h = cv2.boundingRect(pts)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, thickness)

    elif shape == SHAPE_TRIANGLE:
        if approx is not None and len(approx) == 3:
            cv2.drawContours(canvas, [approx], -1, color, thickness)
        elif pts is not None and len(pts) >= 3:
            hull = cv2.convexHull(pts)
            tri = cv2.approxPolyDP(hull, 0.05 * cv2.arcLength(hull, True), True)
            cv2.drawContours(canvas, [tri], -1, color, thickness)

    elif shape in (SHAPE_PENTAGON, SHAPE_HEXAGON):
        if approx is not None:
            cv2.drawContours(canvas, [approx], -1, color, thickness)

    elif shape == SHAPE_LINE:
        if pts is not None and len(pts) > 0:
            pts_2d = np.array(pts, dtype=np.int32).reshape(-1, 2)
            rect = cv2.minAreaRect(pts_2d)
            box  = cv2.boxPoints(rect).astype(np.float32)

            d01 = np.linalg.norm(box[1] - box[0])
            d12 = np.linalg.norm(box[2] - box[1])

            # Determine midpoints of the two shorter edges to span the long axis
            if d01 > d12:
                m1 = (box[1] + box[2]) / 2.0
                m2 = (box[3] + box[0]) / 2.0
            else:
                m1 = (box[0] + box[1]) / 2.0
                m2 = (box[2] + box[3]) / 2.0

            # Preserve user stroke start -> end orientation
            start_user = pts_2d[0]
            if np.linalg.norm(m1 - start_user) > np.linalg.norm(m2 - start_user):
                m1, m2 = m2, m1

            pt1 = (int(round(m1[0])), int(round(m1[1])))
            pt2 = (int(round(m2[0])), int(round(m2[1])))
            cv2.line(canvas, pt1, pt2, color, thickness)

    elif shape == SHAPE_ARROW:
        if approx is not None and len(approx) >= 5:
            cv2.drawContours(canvas, [approx], -1, color, thickness)
        elif pts is not None and len(pts) >= 2:
            pts_2d = np.array(pts, dtype=np.int32).reshape(-1, 2)
            pt_start = tuple(pts_2d[0].tolist())
            pt_end   = tuple(pts_2d[-1].tolist())
            cv2.arrowedLine(canvas, pt_start, pt_end, color, thickness, tipLength=0.25)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_arrow(approx: np.ndarray) -> bool:
    """
    Check if a polygon approximation exhibits arrow characteristics
    (7 vertices with one notably longer edge).
    """
    if approx is None:
        return False
    pts = approx.reshape(-1, 2)
    if len(pts) < 5:
        return False
    edge_lengths = [np.linalg.norm(pts[(i + 1) % len(pts)] - pts[i]) for i in range(len(pts))]
    median = np.median(edge_lengths)
    if median < 1:
        return False
    return max(edge_lengths) / median > 2.2
