"""
AeroScript 2.0 — Hand Tracker
Clean MediaPipe Hands wrapper. Provides a structured landmark dictionary per frame.
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
from config import settings

logger = logging.getLogger(__name__)


class HandTracker:
    """
    Wraps Google MediaPipe Hands with a clean, structured API.

    Output per frame:
        landmarks_px  : dict[int, (px, py)]        — landmark index → pixel coords
        landmarks_norm: dict[int, (nx, ny, nz)]    — normalized [0,1] coords
        world_lm      : list of WorldLandmark       — 3D world coords (metres, wrist=origin)
        handedness    : str "Left" | "Right" | "Unknown"
        all_hands_px  : list[dict[int, (px, py)]]  — all detected hands in pixel coords
        raw           : raw MediaPipe solution result
        raw_hand      : primary hand's landmark list
    """

    # MediaPipe landmark indices (subset used by gesture classifier & UI)
    WRIST        = 0
    THUMB_CMC    = 1
    THUMB_MCP    = 2
    THUMB_IP     = 3
    THUMB_TIP    = 4
    INDEX_MCP    = 5
    INDEX_PIP    = 6
    INDEX_DIP    = 7
    INDEX_TIP    = 8
    MIDDLE_MCP   = 9
    MIDDLE_PIP   = 10
    MIDDLE_DIP   = 11
    MIDDLE_TIP   = 12
    RING_MCP     = 13
    RING_PIP     = 14
    RING_DIP     = 15
    RING_TIP     = 16
    PINKY_MCP    = 17
    PINKY_PIP    = 18
    PINKY_DIP    = 19
    PINKY_TIP    = 20

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._hands    = None
        self._drawing  = mp.solutions.drawing_utils
        self._styles   = mp.solutions.drawing_styles

    def start(self):
        """Initialize MediaPipe Hands session."""
        if self._hands is not None:
            self.stop()

        self._hands = self._mp_hands.Hands(
            max_num_hands           = getattr(settings, "MP_MAX_HANDS", 1),
            min_detection_confidence= getattr(settings, "MP_DETECTION_CONFIDENCE", 0.7),
            min_tracking_confidence = getattr(settings, "MP_TRACKING_CONFIDENCE", 0.6),
        )
        logger.info("MediaPipe Hands initialized.")
        return self

    def stop(self):
        """Close MediaPipe Hands session."""
        if self._hands is not None:
            try:
                self._hands.close()
            except Exception as e:
                logger.warning(f"Error closing MediaPipe hands: {e}")
            finally:
                self._hands = None
                logger.info("MediaPipe Hands closed.")

    def process(self, bgr_frame: np.ndarray) -> dict | None:
        """
        Process one BGR frame and extract hand landmarks.

        Args:
            bgr_frame: OpenCV BGR frame (numpy.ndarray)

        Returns:
            dict with keys:
                'landmarks_px'   : {lm_index: (px, py)}
                'landmarks_norm' : {lm_index: (nx, ny, nz)}
                'world_lm'       : 3D world landmark list | None
                'handedness'     : "Left" | "Right" | "Unknown"
                'all_hands_px'   : [{lm_index: (px, py)}, ...]
                'raw'            : raw MediaPipe result
                'raw_hand'       : primary hand landmarks
            or None if no hand is detected or frame is invalid.
        """
        if bgr_frame is None or not isinstance(bgr_frame, np.ndarray) or bgr_frame.size == 0:
            return None

        if self._hands is None:
            logger.info("HandTracker auto-starting on first process() call.")
            self.start()

        h, w = bgr_frame.shape[:2]
        if h == 0 or w == 0:
            return None

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            result = self._hands.process(rgb)
        except Exception as e:
            logger.error(f"MediaPipe process error: {e}")
            return None

        if not result or not result.multi_hand_landmarks:
            return None

        # Primary hand
        hand_lm = result.multi_hand_landmarks[0]
        handedness = "Unknown"
        if result.multi_handedness and len(result.multi_handedness) > 0:
            classification = result.multi_handedness[0].classification
            if classification and len(classification) > 0:
                handedness = classification[0].label

        lm_px   = {}
        lm_norm = {}
        for idx, lm in enumerate(hand_lm.landmark):
            lm_px[idx]   = (int(round(lm.x * w)), int(round(lm.y * h)))
            lm_norm[idx] = (float(lm.x), float(lm.y), float(lm.z))

        # World landmarks (metres, wrist=origin)
        world_lm = None
        if hasattr(result, "multi_hand_world_landmarks") and result.multi_hand_world_landmarks:
            world_lm = result.multi_hand_world_landmarks[0].landmark

        # All detected hands in pixel space
        all_hands_px = []
        for hl in result.multi_hand_landmarks:
            all_hands_px.append({
                i: (int(round(lm.x * w)), int(round(lm.y * h)))
                for i, lm in enumerate(hl.landmark)
            })

        return {
            "landmarks_px":   lm_px,
            "landmarks_norm": lm_norm,
            "world_lm":       world_lm,
            "handedness":     handedness,
            "all_hands_px":   all_hands_px,
            "raw":            result,
            "raw_hand":       hand_lm,
        }

    def draw_landmarks(self, frame: np.ndarray, hand_data: dict | None, visible: bool = True):
        """
        Draw MediaPipe skeleton on frame (for debug mode or visual feedback).
        """
        if not visible or hand_data is None:
            return
        raw_hand = hand_data.get("raw_hand")
        if raw_hand is None:
            return
        try:
            self._drawing.draw_landmarks(
                frame,
                raw_hand,
                self._mp_hands.HAND_CONNECTIONS,
                self._styles.get_default_hand_landmarks_style(),
                self._styles.get_default_hand_connections_style(),
            )
        except Exception as e:
            logger.debug(f"draw_landmarks error: {e}")

    # ── Static Geometric Helpers ──────────────────────────────────────────────

    @staticmethod
    def is_finger_up(lm_px: dict, tip_idx: int, pip_idx: int) -> bool:
        """Check if finger is pointing up (tip y < pip y)."""
        if tip_idx in lm_px and pip_idx in lm_px:
            return lm_px[tip_idx][1] < lm_px[pip_idx][1]
        return False

    @staticmethod
    def distance(pt1: tuple, pt2: tuple) -> float:
        """Euclidean distance between two 2D points."""
        return float(np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1]))

    # ── Context Manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
