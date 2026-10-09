import cv2
import mediapipe as mp
import numpy as np

# ===== IMPROVED SHAPE DETECTION =====
def detect_shape(contour):
    shape = "Unknown"

    contour = cv2.convexHull(contour)  # smooth + fix gaps

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)

    sides = len(approx)

    if sides == 3:
        shape = "Triangle"

    elif sides == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "Square" if 0.8 <= ar <= 1.2 else "Rectangle"

    elif sides > 4:
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (peri * peri)

        if circularity > 0.6:
            shape = "Circle"

    return shape, approx


mp_hands = mp.solutions.hands
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = None, None

draw_color = (0, 0, 200)
brush_thickness = 5

white_mode = False
eraser_mode = False

prev_pinch_x, prev_pinch_y = None, None
pinch_active = False

drawing_points = []

with mp_hands.Hands(max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    cv2.namedWindow("Hand Painter", cv2.WINDOW_NORMAL)

    while True:
        success, frame = cap.read()
        if not success:
            break

        if cv2.getWindowProperty("Hand Painter", cv2.WND_PROP_VISIBLE) < 1:
            break

        frame = cv2.flip(frame, 1)

        if canvas is None:
            canvas = np.zeros_like(frame)

        white_canvas = np.ones_like(frame) * 255

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        h, w, _ = frame.shape

        color_w = w // 5
        bar_h = 80

        bin_w, bin_h = 100, 60
        bin_x1, bin_y1 = w//2 - bin_w//2, h - bin_h
        bin_x2, bin_y2 = bin_x1 + bin_w, h

        cx, cy = None, None

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                index_tip = hand_landmarks.landmark[8]
                thumb_tip = hand_landmarks.landmark[4]
                index_pip = hand_landmarks.landmark[6]
                middle_tip = hand_landmarks.landmark[12]
                middle_pip = hand_landmarks.landmark[10]

                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

                index_up = index_tip.y < index_pip.y
                middle_up = middle_tip.y < middle_pip.y

                dist = np.hypot(cx - tx, cy - ty)

                # ===== PINCH =====
                if dist < 40:
                    pinch_active = True

                    if prev_pinch_x is not None:
                        dx = cx - prev_pinch_x
                        dy = cy - prev_pinch_y
                        canvas = np.roll(canvas, dx, axis=1)
                        canvas = np.roll(canvas, dy, axis=0)

                    prev_pinch_x, prev_pinch_y = cx, cy

                else:
                    pinch_active = False
                    prev_pinch_x, prev_pinch_y = None, None

                    # ===== SELECTION =====
                    if index_up and middle_up:
                        prev_x, prev_y = None, None
                        drawing_points = []

                        if cy < bar_h:
                            if cx < color_w:
                                draw_color = (0, 0, 200)
                                eraser_mode = False
                            elif cx < 2*color_w:
                                draw_color = (0, 200, 0)
                                eraser_mode = False
                            elif cx < 3*color_w:
                                draw_color = (200, 0, 0)
                                eraser_mode = False
                            elif cx < 4*color_w:
                                eraser_mode = True
                            else:
                                white_mode = not white_mode

                    # ===== DRAW =====
                    elif index_up:
                        drawing_points.append((cx, cy))

                        if prev_x is not None:
                            dx = cx - prev_x
                            dy = cy - prev_y
                            dist = int(np.hypot(dx, dy))
                            if dist == 0:
                                dist = 1

                            for i in range(dist):
                                x = int(prev_x + dx * i / dist)
                                y = int(prev_y + dy * i / dist)

                                if eraser_mode:
                                    cv2.circle(canvas, (x, y), 30, (0, 0, 0), -1)
                                else:
                                    cv2.circle(canvas, (x, y), brush_thickness, draw_color, -1)

                        prev_x, prev_y = cx, cy

                    else:
                        # ===== FAST + ROBUST SHAPE DETECTION =====
                        if len(drawing_points) > 25:

                            pts = np.array(drawing_points, dtype=np.int32)

                            # auto-close shape
                            pts = np.vstack([pts, pts[0]])

                            contour = pts.reshape((-1, 1, 2))

                            shape, approx = detect_shape(contour)

                            # erase rough
                            for p in drawing_points:
                                cv2.circle(canvas, p, brush_thickness+2, (0, 0, 0), -1)

                            # draw perfect
                            if shape == "Circle":
                                (x, y), r = cv2.minEnclosingCircle(contour)
                                cv2.circle(canvas, (int(x), int(y)), int(r), draw_color, 3)

                            elif shape in ["Rectangle", "Square"]:
                                x, y, w, h = cv2.boundingRect(contour)
                                cv2.rectangle(canvas, (x, y), (x+w, y+h), draw_color, 3)

                            elif shape == "Triangle":
                                cv2.drawContours(canvas, [approx], -1, draw_color, 3)

                        drawing_points = []
                        prev_x, prev_y = None, None

        else:
            prev_x, prev_y = None, None
            drawing_points = []

        # ===== BIN DELETE =====
        if pinch_active and cx is not None:
            if bin_x1 < cx < bin_x2 and bin_y1 < cy < bin_y2:
                canvas = np.zeros_like(frame)

        # ===== MERGE =====
        bg = white_canvas if white_mode else frame

        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        combined = cv2.add(
            cv2.bitwise_and(bg, bg, mask=mask_inv),
            cv2.bitwise_and(canvas, canvas, mask=mask)
        )

        # ===== UI =====
        cv2.rectangle(combined, (0, 0), (color_w, bar_h), (0, 0, 200), -1)
        cv2.rectangle(combined, (color_w, 0), (2*color_w, bar_h), (0, 200, 0), -1)
        cv2.rectangle(combined, (2*color_w, 0), (3*color_w, bar_h), (200, 0, 0), -1)

        cv2.rectangle(combined, (3*color_w, 0), (4*color_w, bar_h), (50, 50, 50), -1)
        cv2.circle(combined, (3*color_w + color_w//2, bar_h//2), 25, (255,255,255), -1)

        cv2.rectangle(combined, (4*color_w, 0), (5*color_w, bar_h), (80, 80, 80), -1)
        cv2.rectangle(combined,
                      (4*color_w+20, 20),
                      (5*color_w-20, bar_h-20),
                      (255,255,255), 3)

        if pinch_active:
            cv2.rectangle(combined, (bin_x1, bin_y1), (bin_x2, bin_y2), (0,0,255), -1)
            cv2.putText(combined, "BIN", (bin_x1+20, bin_y1+40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if cx is not None:
            cv2.circle(combined, (cx, cy), 10, (0,0,0), -1)
            cv2.circle(combined, (cx, cy), 6, (0,255,255), -1)

        cv2.imshow("Hand Painter", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()