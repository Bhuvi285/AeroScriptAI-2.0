"""
AeroScript 2.0 — Air-Writing & Spatial Character Segmentation Pipeline
=====================================================================
Segments hand-drawn air-writing strokes into discrete characters and 
runs high-accuracy PyTorch CNN inference to convert air-writing to text words 
and mathematical expressions.

Key Steps:
1. Canvas ink binarization & contour detection
2. Multi-stroke clustering (merges disjoint strokes like '=', '+', 'i', 'H')
3. Left-to-right natural reading order sorting & space detection
4. Aspect-ratio preserving 28x28 normalization
5. PyTorch CNN inference with mode-aware contextual disambiguation
"""

import cv2
import numpy as np
import logging
from ai.character_recognizer import CharacterClassifier
from ai.math_engine import MathEngine

log = logging.getLogger("AeroScript.AirWriting")


class AirWritingEngine:
    """
    Core engine for air-writing recognition and smart math solving.
    """

    def __init__(self):
        self.classifier = CharacterClassifier.get_instance()

    def process_canvas(self, surface: np.ndarray, mode: str = "text") -> dict:
        """
        Extracts, segments, and recognizes characters drawn on the canvas surface.

        Args:
            surface: BGR numpy array representing the canvas drawing surface.
            mode: "text" for air-writing to text, "math" for mathematical solver.

        Returns:
            dict containing:
                "text": str,
                "confidence": float,
                "characters": list of dicts with {"char", "box", "conf"},
                "math_solution": dict (if mode == "math"),
                "boxes": list of (x, y, w, h)
        """
        if surface is None or surface.size == 0:
            return self._empty_result(mode)

        # 1. Binarize canvas ink
        if len(surface.shape) == 3:
            gray = cv2.cvtColor(surface, cv2.COLOR_BGR2GRAY)
        else:
            gray = surface.copy()

        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Check if there is any drawing
        if cv2.countNonZero(binary) < 40:
            return self._empty_result(mode)

        # 2. Extract initial contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        initial_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter tiny noise
            if w * h >= 30 and (w >= 4 or h >= 8):
                initial_boxes.append((x, y, w, h))

        if not initial_boxes:
            return self._empty_result(mode)

        # 3. Merge multi-stroke components (e.g. '=', '+', 'H', dots)
        merged_boxes = self._merge_character_boxes(initial_boxes)

        # 4. Sort left-to-right
        merged_boxes.sort(key=lambda b: b[0])

        # 5. Extract glyphs & predict
        char_results = []
        glyphs_28 = []

        for (bx, by, bw, bh) in merged_boxes:
            crop = binary[by:by+bh, bx:bx+bw]
            norm_28 = self._normalize_glyph(crop, bw, bh)
            glyphs_28.append(norm_28)

        # Batch predict
        predictions = self.classifier.predict_batch(glyphs_28)

        # 6. Disambiguation and string assembly
        assembled_chars = []
        total_conf = 0.0

        for i, ((bx, by, bw, bh), (pred_char, conf)) in enumerate(zip(merged_boxes, predictions)):
            # Disambiguate based on mode and geometric features
            refined_char = self._disambiguate(pred_char, bw, bh, mode)
            assembled_chars.append(refined_char)
            total_conf += conf
            char_results.append({
                "char": refined_char,
                "box": (bx, by, bw, bh),
                "conf": conf
            })

        avg_conf = (total_conf / len(char_results)) if char_results else 0.0
        full_text = "".join(assembled_chars)

        # Add spaces if gap between characters is large
        if mode == "text" and len(merged_boxes) > 1:
            full_text_spaced = self._insert_word_spaces(char_results)
        else:
            full_text_spaced = full_text

        result = {
            "text": full_text_spaced,
            "confidence": avg_conf,
            "characters": char_results,
            "boxes": merged_boxes,
            "math_solution": None
        }

        # 7. If math mode, evaluate expression
        if mode == "math":
            math_res = MathEngine.evaluate(full_text)
            result["math_solution"] = math_res
            if math_res["valid"]:
                result["text"] = math_res["formatted"]

        return result

    def _normalize_glyph(self, crop: np.ndarray, bw: int, bh: int) -> np.ndarray:
        """
        Normalizes a cropped binary ink glyph into a centered 28x28 grayscale image.
        Maintains aspect ratio and pads with black borders.
        """
        max_dim = max(bw, bh)
        scale = 20.0 / max(max_dim, 1)
        nw = max(int(bw * scale), 1)
        nh = max(int(bh * scale), 1)

        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

        # Center in 28x28
        norm = np.zeros((28, 28), dtype=np.uint8)
        ox = (28 - nw) // 2
        oy = (28 - nh) // 2
        norm[oy:oy+nh, ox:ox+nw] = resized
        return norm

    def _merge_character_boxes(self, boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """
        Merges disjoint strokes belonging to the same character:
        - Vertically stacked strokes with strong horizontal overlap (e.g. '=' or 'i' dot)
        - Close intersecting/crossing strokes (e.g. '+' cross strokes or 'H' crossbars)
        Prevents merging horizontally separated adjacent characters.
        """
        if len(boxes) <= 1:
            return boxes

        merged = list(boxes)
        changed = True

        while changed:
            changed = False
            new_merged = []
            used = [False] * len(merged)

            for i in range(len(merged)):
                if used[i]:
                    continue
                x1, y1, w1, h1 = merged[i]
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    x2, y2, w2, h2 = merged[j]

                    # Overlap in X and Y
                    overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                    overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                    dist_x    = max(0, max(x1, x2) - min(x1 + w1, x2 + w2))
                    dist_y    = max(0, max(y1, y2) - min(y1 + h1, y2 + h2))

                    min_w = min(w1, w2)
                    max_h = max(h1, h2)

                    # 1. Vertically stacked strokes (e.g. '=' or 'i'/'j' dot):
                    # High horizontal overlap (one directly above the other) and small vertical gap
                    is_vertical_stack = (overlap_x >= 0.40 * min_w) and (dist_y <= max_h * 1.8) and (dist_x == 0)

                    # 2. Direct intersection / crossing strokes (e.g. '+' cross strokes or 'A'/'H' crossbar)
                    is_intersecting = (overlap_x > 0 and overlap_y > 0)

                    if is_vertical_stack or is_intersecting:
                        nx = min(x1, x2)
                        ny = min(y1, y2)
                        nw = max(x1 + w1, x2 + w2) - nx
                        nh = max(y1 + h1, y2 + h2) - ny
                        x1, y1, w1, h1 = nx, ny, nw, nh
                        used[j] = True
                        changed = True

                new_merged.append((x1, y1, w1, h1))
                used[i] = True

            merged = new_merged

        return merged

    def _disambiguate(self, char: str, w: int, h: int, mode: str) -> str:
        """
        Disambiguates visually identical glyphs depending on context and aspect ratio.
        """
        ar = w / float(h + 1e-6)

        if mode == "math":
            # In math:
            if char in ('O', 'o', 'D'):
                return '0'
            if char in ('I', 'l', '|'):
                return '1'
            if char in ('S', 's'):
                return '5'
            if char in ('Z', 'z'):
                return '2'
            if char in ('X', 'x'):
                return '*'
            if char in ('T', 't'):
                # Check aspect ratio: '+' is symmetric (ar ~ 1.0)
                return '+' if 0.6 <= ar <= 1.4 else '1'
            if char == '-':
                return '-'
        elif mode == "text":
            # In text:
            if char == '0':
                return 'O'
            if char == '1':
                return 'I'
            if char == '5':
                return 'S'
            if char == '+':
                return 'T'

        return char

    def _insert_word_spaces(self, char_results: list[dict]) -> str:
        """
        Inserts space characters between words based on horizontal gaps.
        """
        if not char_results:
            return ""

        avg_w = sum(c["box"][2] for c in char_results) / len(char_results)
        threshold_gap = max(avg_w * 0.95, 25)

        text_out = []
        for i, cr in enumerate(char_results):
            if i > 0:
                prev_right = char_results[i-1]["box"][0] + char_results[i-1]["box"][2]
                curr_left  = cr["box"][0]
                gap = curr_left - prev_right
                if gap > threshold_gap:
                    text_out.append(" ")
            text_out.append(cr["char"])

        return "".join(text_out)

    def _empty_result(self, mode: str) -> dict:
        return {
            "text": "",
            "confidence": 0.0,
            "characters": [],
            "boxes": [],
            "math_solution": None
        }
