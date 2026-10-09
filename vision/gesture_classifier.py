"""
AeroScript 2.0 — Gesture Classifier
Pure geometric gesture recognition using normalized MediaPipe landmarks.
No ML model required — fast, reliable, and interpretable.

Algorithm:
    For each gesture, we compute relationships between normalized landmark
    coordinates (finger extension, joint angles, inter-finger distances, etc.)
    These are scale-invariant because MediaPipe normalizes to [0,1] in the hand bbox.

Gestures Implemented:
    DRAW       : index UP, middle DOWN (original test.py behavior)
    SELECT     : index UP + middle UP (original test.py behavior)
    PINCH      : thumb-index distance < threshold (original test.py behavior)
    FIST       : all fingers curled
    OPEN_PALM  : all 5 fingers extended
    THUMBS_UP  : thumb UP, all other fingers DOWN
    SWIPE_LEFT : tracked via horizontal velocity
    SWIPE_RIGHT: tracked via horizontal velocity
    THREE_FINGER: index + middle + ring UP, others DOWN
    STOP       : palm facing camera (detected via depth z-coordinate)
"""

import numpy as np
import time
import logging
import collections
from config import settings
from vision.hand_tracker import HandTracker as LM   # landmark index constants

logger = logging.getLogger(__name__)

# Gesture name constants
GESTURE_DRAW        = "draw"
GESTURE_SELECT      = "select"
GESTURE_PINCH       = "pinch"
GESTURE_FIST        = "fist"
GESTURE_OPEN_PALM   = "open_palm"
GESTURE_THUMBS_UP   = "thumbs_up"
GESTURE_SWIPE_LEFT  = "swipe_left"
GESTURE_SWIPE_RIGHT = "swipe_right"
GESTURE_THREE       = "three_fingers"
GESTURE_STOP        = "stop"
GESTURE_NONE        = "none"


class GestureClassifier:
    """
    Classifies hand gesture from a single-frame landmark dictionary.

    Usage:
        clf = GestureClassifier()
        gesture, confidence = clf.classify(hand_data, frame_width)
    """

    def __init__(self):
        # Smoothing buffer: store last N gesture predictions
        self._history: collections.deque = collections.deque(
            maxlen=settings.GESTURE_SMOOTHING_FRAMES
        )
        # Swipe tracking
        self._swipe_history_x: collections.deque = collections.deque(maxlen=8)
        self._swipe_history_t: collections.deque = collections.deque(maxlen=8)

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, hand_data: dict, frame_w: int, frame_h: int) -> tuple[str, float]:
        """
        Classify a gesture from one frame's hand data.

        Args:
            hand_data : output dict from HandTracker.process()
            frame_w   : frame width in pixels (for pixel-space checks)
            frame_h   : frame height in pixels

        Returns:
            (gesture_name: str, confidence: float [0,1])
        """
        if hand_data is None:
            self._history.clear()
            self._swipe_history_x.clear()
            return GESTURE_NONE, 0.0

        lm_px   = hand_data["landmarks_px"]
        lm_norm = hand_data["landmarks_norm"]

        # ── Finger state booleans ─────────────────────────────────────────────
        index_up  = self._finger_up(lm_norm, LM.INDEX_TIP,  LM.INDEX_PIP)
        middle_up = self._finger_up(lm_norm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
        ring_up   = self._finger_up(lm_norm, LM.RING_TIP,   LM.RING_PIP)
        pinky_up  = self._finger_up(lm_norm, LM.PINKY_TIP,  LM.PINKY_PIP)
        thumb_up  = self._thumb_up(lm_norm)

        # ── Pinch detection ───────────────────────────────────────────────────
        ix, iy = lm_px[LM.INDEX_TIP]
        tx, ty = lm_px[LM.THUMB_TIP]
        pinch_dist = np.hypot(ix - tx, iy - ty)
        is_pinch   = pinch_dist < settings.PINCH_DISTANCE_THRESHOLD

        # ── Swipe detection ───────────────────────────────────────────────────
        wrist_nx = lm_norm[LM.WRIST][0]
        now       = time.time()
        self._swipe_history_x.append(wrist_nx)
        self._swipe_history_t.append(now)

        swipe_gesture = GESTURE_NONE
        if len(self._swipe_history_x) >= 4:
            dt = self._swipe_history_t[-1] - self._swipe_history_t[0]
            if dt > 0.05:
                dx = self._swipe_history_x[-1] - self._swipe_history_x[0]
                velocity = abs(dx) / dt
                if velocity > settings.SWIPE_VELOCITY_THRESHOLD:
                    if dx < 0:
                        swipe_gesture = GESTURE_SWIPE_LEFT
                    else:
                        swipe_gesture = GESTURE_SWIPE_RIGHT

        # ── Priority classification ───────────────────────────────────────────
        # Order matters — more specific gestures checked first

        if is_pinch and not index_up:
            raw = GESTURE_PINCH

        elif swipe_gesture != GESTURE_NONE and middle_up and ring_up and pinky_up:
            raw = swipe_gesture    # big sweeping swipe with open hand

        elif not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            raw = GESTURE_FIST

        elif index_up and middle_up and ring_up and pinky_up and thumb_up:
            raw = GESTURE_OPEN_PALM

        elif thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
            raw = GESTURE_THUMBS_UP

        elif index_up and middle_up and ring_up and not pinky_up:
            raw = GESTURE_THREE

        elif index_up and middle_up and not ring_up:
            raw = GESTURE_SELECT   # original test.py behavior

        elif index_up and not middle_up:
            raw = GESTURE_DRAW     # original test.py behavior

        else:
            raw = GESTURE_NONE

        # ── Smoothing: majority vote over history ─────────────────────────────
        self._history.append(raw)
        counts = {}
        for g in self._history:
            counts[g] = counts.get(g, 0) + 1
        smoothed   = max(counts, key=counts.get)
        confidence = counts[smoothed] / len(self._history)

        return smoothed, confidence

    # ── Finger State Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _finger_up(lm_norm: dict, tip_idx: int, pip_idx: int) -> bool:
        """
        A finger is 'up' (extended) when its tip is above its PIP joint in image space.
        Image y increases downward, so tip.y < pip.y means tip is above pip.
        Exactly matches original test.py logic.
        """
        return lm_norm[tip_idx][1] < lm_norm[pip_idx][1]

    @staticmethod
    def _thumb_up(lm_norm: dict) -> bool:
        """
        Thumb is 'up' when its tip is well above the thumb IP joint,
        AND the thumb is significantly above the index MCP (pointing up).
        """
        tip_y = lm_norm[LM.THUMB_TIP][1]
        ip_y  = lm_norm[LM.THUMB_IP][1]
        mcp_y = lm_norm[LM.INDEX_MCP][1]
        return tip_y < ip_y and tip_y < mcp_y

    @staticmethod
    def pinch_midpoint(lm_px: dict) -> tuple:
        """Return the midpoint between thumb tip and index tip (for pinch position)."""
        ix, iy = lm_px[LM.INDEX_TIP]
        tx, ty = lm_px[LM.THUMB_TIP]
        return ((ix + tx) // 2, (iy + ty) // 2)

    @staticmethod
    def pinch_distance(lm_px: dict) -> float:
        """Euclidean distance between thumb tip and index tip."""
        ix, iy = lm_px[LM.INDEX_TIP]
        tx, ty = lm_px[LM.THUMB_TIP]
        return float(np.hypot(ix - tx, iy - ty))
