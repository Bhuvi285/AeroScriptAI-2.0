"""
AeroScript 2.0 — Camera Module
Handles webcam initialization, frame reading with error recovery, and resolution configuration.
"""

import cv2
import logging
import time
from config import settings

logger = logging.getLogger(__name__)


class Camera:
    """
    Manages the webcam lifecycle with graceful error recovery.

    Features:
    - Configurable resolution and FPS
    - Auto-reconnect on frame drop
    - Horizontal flip (selfie-mirror mode)
    """

    def __init__(self,
                 index: int = settings.CAMERA_INDEX,
                 width: int = settings.CAMERA_WIDTH,
                 height: int = settings.CAMERA_HEIGHT,
                 fps: int = settings.CAMERA_FPS,
                 flip: bool = settings.CAMERA_FLIP):
        self.index  = index
        self.width  = width
        self.height = height
        self.fps    = fps
        self.flip   = flip
        self._cap   = None
        self._consecutive_failures = 0
        self._max_failures         = 10

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> bool:
        """
        Open the webcam. Returns True on success, False if camera unavailable.
        """
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            logger.error(f"Camera {self.index} could not be opened.")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.fps)

        self._consecutive_failures = 0
        logger.info(f"Camera {self.index} opened at "
                    f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                    f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                    f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))} FPS")
        return True

    def release(self):
        """Release the camera resource."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Camera released.")

    # ── Frame Reading ─────────────────────────────────────────────────────────

    def read(self):
        """
        Read one frame. Applies horizontal flip if enabled.

        Returns:
            (success: bool, frame: np.ndarray | None)

        On consecutive failures, attempts one reconnect before giving up.
        """
        if self._cap is None or not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()

        if not ret:
            self._consecutive_failures += 1
            logger.warning(f"Frame read failure #{self._consecutive_failures}")

            if self._consecutive_failures >= self._max_failures:
                logger.warning("Too many failures — attempting camera reconnect.")
                self._reconnect()

            return False, None

        self._consecutive_failures = 0

        if self.flip:
            frame = cv2.flip(frame, 1)

        return True, frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def get_resolution(self) -> tuple:
        if not self.is_open():
            return (self.width, self.height)
        return (
            int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def _reconnect(self):
        """Release and re-open the camera after failures."""
        self.release()
        time.sleep(0.5)
        success = self.open()
        if success:
            logger.info("Camera reconnected successfully.")
        else:
            logger.error("Camera reconnect failed.")

    # ── Context Manager Support ───────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.release()
