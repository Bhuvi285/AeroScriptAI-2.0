"""
AeroScript 2.0 — Canvas Module
Manages the drawing surface with undo/redo history, pan, and clear operations.
"""

import cv2
import numpy as np
import collections
import logging
from config import settings

logger = logging.getLogger(__name__)


class Canvas:
    """
    Manages the drawing canvas (numpy array) with full undo/redo support.

    Design:
    - The canvas is a numpy array (H, W, 3) of dtype uint8, all zeros (black = transparent).
    - Undo/Redo uses a deque of canvas snapshots (memory-efficient: only stores deltas
      would be ideal, but snapshots at ~720p are ~2.7 MB each × 50 = ~135 MB max — acceptable).
    - Pan uses np.roll, exactly as in the original test.py.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width

        self._surface  = np.zeros((height, width, 3), dtype=np.uint8)
        self._undo_stack: collections.deque = collections.deque(maxlen=settings.CANVAS_UNDO_LIMIT)
        self._redo_stack: collections.deque = collections.deque(maxlen=settings.CANVAS_UNDO_LIMIT)

    # ── Surface Access ────────────────────────────────────────────────────────

    @property
    def surface(self) -> np.ndarray:
        """Return the current drawing surface (read-write)."""
        return self._surface

    def resize(self, height: int, width: int):
        """Resize the canvas, preserving existing content (cropped or padded)."""
        new = np.zeros((height, width, 3), dtype=np.uint8)
        h = min(height, self.height)
        w = min(width,  self.width)
        new[:h, :w] = self._surface[:h, :w]
        self._surface = new
        self.height, self.width = height, width

    # ── Undo / Redo ───────────────────────────────────────────────────────────

    def save_snapshot(self):
        """
        Save the current canvas state to the undo stack.
        Call this BEFORE making a drawing change (stroke start, shape finalize, clear, etc.).
        Clears the redo stack because a new action invalidates future redo history.
        """
        self._undo_stack.append(self._surface.copy())
        self._redo_stack.clear()

    def undo(self) -> bool:
        """
        Revert to the previous state.

        Returns:
            True if undo was performed, False if nothing to undo.
        """
        if not self._undo_stack:
            logger.debug("Undo stack empty — nothing to undo.")
            return False
        self._redo_stack.append(self._surface.copy())
        self._surface = self._undo_stack.pop()
        self._mark_dirty()
        logger.debug(f"Undo performed. Stack size: {len(self._undo_stack)}")
        return True

    def redo(self) -> bool:
        """
        Re-apply the last undone state.

        Returns:
            True if redo was performed, False if nothing to redo.
        """
        if not self._redo_stack:
            logger.debug("Redo stack empty — nothing to redo.")
            return False
        self._undo_stack.append(self._surface.copy())
        self._surface = self._redo_stack.pop()
        self._mark_dirty()
        logger.debug(f"Redo performed. Stack size: {len(self._redo_stack)}")
        return True

    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    # ── Operations ────────────────────────────────────────────────────────────

    def clear(self):
        """Clear the canvas (save snapshot for undo first)."""
        self.save_snapshot()
        self._surface[:] = 0
        self._mark_dirty()
        logger.debug("Canvas cleared.")

    def pan(self, dx: int, dy: int):
        """
        Translate the canvas contents by (dx, dy) pixels using np.roll.
        Migrated directly from original test.py — preserves existing behavior.
        """
        self._surface = np.roll(self._surface, dx, axis=1)
        self._surface = np.roll(self._surface, dy, axis=0)
        self._mark_dirty()

    # ── Serialization (for save/load) ─────────────────────────────────────────

    def to_array(self) -> np.ndarray:
        """Return a copy of the canvas surface for saving."""
        return self._surface.copy()

    def from_array(self, arr: np.ndarray):
        """Load a canvas from a numpy array."""
        if arr.shape != self._surface.shape:
            # Resize if needed
            arr = np.resize(arr, self._surface.shape)
        self.save_snapshot()
        self._surface = arr.copy().astype(np.uint8)
        self._mark_dirty()

    # ── Mask Compositing ──────────────────────────────────────────────────────

    def composite(self, background: np.ndarray) -> np.ndarray:
        """
        Composite the canvas drawing layer over a background frame.
        Uses cached mask to avoid recomputing cvtColor + threshold every frame
        when the canvas hasn't been modified.

        Args:
            background: BGR frame (webcam or white canvas)

        Returns:
            Combined BGR image
        """
        if getattr(self, '_mask_dirty', True):
            gray = cv2.cvtColor(self._surface, cv2.COLOR_BGR2GRAY)
            _, self._mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
            self._mask_inv = cv2.bitwise_not(self._mask)
            self._mask_dirty = False

        result = cv2.add(
            cv2.bitwise_and(background, background, mask=self._mask_inv),
            cv2.bitwise_and(self._surface, self._surface, mask=self._mask)
        )
        return result

    def _mark_dirty(self):
        """Mark the canvas mask as needing recomputation."""
        self._mask_dirty = True

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    @property
    def redo_depth(self) -> int:
        return len(self._redo_stack)

    def pixel_coverage(self) -> float:
        """Fraction of canvas pixels that have any drawing on them."""
        gray = cv2.cvtColor(self._surface, cv2.COLOR_BGR2GRAY)
        non_zero = np.count_nonzero(gray)
        return non_zero / (self.height * self.width)
