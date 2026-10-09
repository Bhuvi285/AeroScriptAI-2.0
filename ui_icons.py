import cv2
import numpy as np

def draw_icon(out: np.ndarray, name: str, cx: int, cy: int, size: int, color: tuple, thickness: int = 2):
    """
    Draws a scalable vector icon using anti-aliased OpenCV primitives.
    cx, cy: Center coordinates of the icon.
    size: Approximate bounding box size of the icon.
    """
    r = size // 2
    AA = cv2.LINE_AA
    
    if name == "pen":
        # Diagonal sleek pen: bottom-left nib to top-right barrel
        p1 = (cx - r + 3, cy + r - 3)
        p2 = (cx + r - 3, cy - r + 3)
        cv2.line(out, p1, p2, color, thickness + 1, AA)
        # Nib triangle
        nib = np.array([
            [cx - r + 1, cy + r - 1],
            [cx - r + 6, cy + r - 1],
            [cx - r + 1, cy + r - 6]
        ], np.int32)
        cv2.fillPoly(out, [nib], color, AA)
        # Collar accent
        cv2.line(out, (cx, cy), (cx + 2, cy - 2), (255, 255, 255), thickness, AA)

    elif name == "marker":
        # Thick angled marker body
        p1 = (cx - r + 3, cy + r - 3)
        p2 = (cx + r - 3, cy - r + 3)
        cv2.line(out, p1, p2, color, thickness + 3, AA)
        # Flat chisel tip
        cv2.line(out, (cx - r + 1, cy + r - 4), (cx - r + 4, cy + r - 1), color, thickness + 1, AA)

    elif name == "highlighter":
        # Wide chisel tip highlighter
        p1 = (cx - r + 3, cy + r - 3)
        p2 = (cx + r - 3, cy - r + 3)
        cv2.line(out, p1, p2, color, thickness + 5, AA)
        # Rectangular cap collar
        cv2.line(out, (cx + r - 5, cy - r + 1), (cx + r - 1, cy - r + 5), (255, 255, 255), thickness, AA)

    elif name == "eraser":
        # Angled modern eraser block
        pts = np.array([
            [cx - r + 2, cy + 2],
            [cx - 2, cy - r + 2],
            [cx + r - 2, cy - 2],
            [cx + 2, cy + r - 2]
        ], np.int32)
        cv2.polylines(out, [pts], True, color, thickness, AA)
        # Division band across eraser
        cv2.line(out, (cx - r//2 + 1, cy - r//2 + 3), (cx + r//2 - 1, cy + r//2 - 1), color, thickness, AA)

    elif name == "smart_math":
        # Greek Sigma symbol ∑
        pts = np.array([
            [cx + r - 2, cy - r + 3],
            [cx - r + 3, cy - r + 3],
            [cx + 1, cy],
            [cx - r + 3, cy + r - 3],
            [cx + r - 2, cy + r - 3]
        ], np.int32)
        cv2.polylines(out, [pts], False, color, thickness, AA)

    elif name == "air_text":
        # Modern typographic 'T' with subtle serif bars
        cv2.line(out, (cx - r + 2, cy - r + 3), (cx + r - 2, cy - r + 3), color, thickness, AA)
        cv2.line(out, (cx, cy - r + 3), (cx, cy + r - 3), color, thickness, AA)
        cv2.line(out, (cx - 3, cy + r - 3), (cx + 3, cy + r - 3), color, thickness, AA)

    elif name == "clear":
        # Modern waste bin icon
        cv2.rectangle(out, (cx - r + 3, cy - r + 5), (cx + r - 3, cy + r - 2), color, thickness, AA)
        # Lid & handle
        cv2.line(out, (cx - r + 1, cy - r + 5), (cx + r - 1, cy - r + 5), color, thickness, AA)
        cv2.line(out, (cx - 3, cy - r + 2), (cx + 3, cy - r + 2), color, thickness, AA)
        # Inner vertical slats
        cv2.line(out, (cx - 2, cy - r + 8), (cx - 2, cy + r - 5), color, 1, AA)
        cv2.line(out, (cx + 2, cy - r + 8), (cx + 2, cy + r - 5), color, 1, AA)

    elif name == "save":
        # Floppy diskette / save icon
        cv2.rectangle(out, (cx - r + 2, cy - r + 2), (cx + r - 2, cy + r - 2), color, thickness, AA)
        cv2.rectangle(out, (cx - r + 5, cy - r + 2), (cx + r - 5, cy - 2), color, 1, AA)
        cv2.rectangle(out, (cx - r + 4, cy + 2), (cx + r - 4, cy + r - 2), color, 1, AA)

    elif name == "gemini":
        # Sparkle 4-point star
        pts = np.array([
            [cx, cy - r + 1],
            [cx + 3, cy - 3],
            [cx + r - 1, cy],
            [cx + 3, cy + 3],
            [cx, cy + r - 1],
            [cx - 3, cy + 3],
            [cx - r + 1, cy],
            [cx - 3, cy - 3]
        ], np.int32)
        cv2.fillPoly(out, [pts], color, AA)

    elif name == "whiteboard":
        # Screen board with sleek tripod
        cv2.rectangle(out, (cx - r + 2, cy - r + 2), (cx + r - 2, cy + 3), color, thickness, AA)
        cv2.line(out, (cx - r + 3, cy + 3), (cx - r - 1, cy + r - 1), color, thickness, AA)
        cv2.line(out, (cx + r - 3, cy + 3), (cx + r + 1, cy + r - 1), color, thickness, AA)
        cv2.line(out, (cx, cy + 3), (cx, cy + r - 1), color, 1, AA)

    elif name == "camera":
        # Camera icon
        cv2.rectangle(out, (cx - r + 2, cy - r + 4), (cx + r - 2, cy + r - 2), color, thickness, AA)
        cv2.circle(out, (cx, cy + 1), r - 4, color, thickness, AA)
        cv2.circle(out, (cx + r - 5, cy - r + 7), 1, color, -1, AA)
        # Top viewfinder bump
        cv2.line(out, (cx - 3, cy - r + 4), (cx + 3, cy - r + 4), color, thickness, AA)

    elif name == "increase_brush":
        cv2.circle(out, (cx, cy), r - 1, color, 1, AA)
        cv2.line(out, (cx - r + 5, cy), (cx + r - 5, cy), color, thickness, AA)
        cv2.line(out, (cx, cy - r + 5), (cx, cy + r - 5), color, thickness, AA)

    elif name == "decrease_brush":
        cv2.circle(out, (cx, cy), r - 1, color, 1, AA)
        cv2.line(out, (cx - r + 5, cy), (cx + r - 5, cy), color, thickness, AA)

    elif name == "hamburger":
        # Clean 3-line hamburger menu
        cv2.line(out, (cx - r + 3, cy - r + 4), (cx + r - 3, cy - r + 4), color, thickness, AA)
        cv2.line(out, (cx - r + 3, cy),         (cx + r - 3, cy),         color, thickness, AA)
        cv2.line(out, (cx - r + 3, cy + r - 4), (cx + r - 3, cy + r - 4), color, thickness, AA)

    elif name == "shape_pen":
        # Modern overlapping geometric primitives (square + circle)
        cv2.rectangle(out, (cx - r + 2, cy - r + 2), (cx + 2, cy + 2), color, thickness, AA)
        cv2.circle(out, (cx + 2, cy + 2), r - 3, color, thickness, AA)

    elif name == "mic":
        # Scalable capsule microphone
        mic_w = max(r // 3, 3)
        mic_h = r - 1
        cv2.rectangle(out, (cx - mic_w, cy - mic_h), (cx + mic_w, cy + 1), color, -1, AA)
        cv2.circle(out, (cx, cy - mic_h), mic_w, color, -1, AA)
        cv2.circle(out, (cx, cy + 1), mic_w, color, -1, AA)
        # Stand cradle
        cv2.ellipse(out, (cx, cy + 1), (mic_w + 3, mic_w + 3), 0, 0, 180, color, thickness, AA)
        cv2.line(out, (cx, cy + mic_w + 4), (cx, cy + mic_w + 8), color, thickness, AA)
        cv2.line(out, (cx - 4, cy + mic_w + 8), (cx + 4, cy + mic_w + 8), color, thickness, AA)

    elif name == "close":
        cv2.line(out, (cx - r + 4, cy - r + 4), (cx + r - 4, cy + r - 4), color, thickness, AA)
        cv2.line(out, (cx + r - 4, cy - r + 4), (cx - r + 4, cy + r - 4), color, thickness, AA)

    elif name == "help":
        cv2.ellipse(out, (cx, cy - r // 2), (r - 3, r - 3), 0, 180, 360, color, thickness, AA)
        cv2.line(out, (cx + r - 3, cy - r // 2), (cx, cy + 2), color, thickness, AA)
        cv2.circle(out, (cx, cy + r - 3), 2, color, -1, AA)

    elif name == "undo":
        cv2.ellipse(out, (cx, cy + 1), (r - 3, r - 3), 0, 150, 360, color, thickness, AA)
        cv2.line(out, (cx - r + 3, cy - 2), (cx - r + 3, cy - r + 3), color, thickness, AA)
        cv2.line(out, (cx - r + 3, cy - 2), (cx - 2, cy - 2), color, thickness, AA)

    elif name == "redo":
        cv2.ellipse(out, (cx, cy + 1), (r - 3, r - 3), 0, 180, 390, color, thickness, AA)
        cv2.line(out, (cx + r - 3, cy - 2), (cx + r - 3, cy - r + 3), color, thickness, AA)
        cv2.line(out, (cx + r - 3, cy - 2), (cx + 2, cy - 2), color, thickness, AA)

    elif name == "check":
        # Checkmark
        cv2.line(out, (cx - r + 3, cy), (cx - 1, cy + r - 3), color, thickness, AA)
        cv2.line(out, (cx - 1, cy + r - 3), (cx + r - 2, cy - r + 3), color, thickness, AA)

    elif name == "waveform":
        # Soundwave bars
        bars = [-3, -6, -2, -8, -1, -5, -2]
        step = max(size // 8, 2)
        start_x = cx - (len(bars) * step) // 2
        for i, b in enumerate(bars):
            bx = start_x + i * step
            cv2.line(out, (bx, cy - abs(b)), (bx, cy + abs(b)), color, thickness, AA)

    else:
        # Default smooth circle dot
        cv2.circle(out, (cx, cy), r - 2, color, thickness, AA)
