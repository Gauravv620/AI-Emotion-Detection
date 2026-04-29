import cv2
import mediapipe as mp
import math
import random
import numpy as np


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


hearts = []

def add_heart(x, y):
    hearts.append({
        "x": int(x),
        "y": int(y),
        "size": random.randint(14, 22),
        "speed": random.randint(2, 4)
    })

def draw_heart(frame, x, y, size):
    # FORCE INT (CRITICAL FOR OPENCV)
    x = int(x)
    y = int(y)
    size = int(size)

    pink = (203, 192, 255)   # BGR pink
    yellow = (0, 255, 255)   # star color

    # --- HEART SHAPE ---
    # Left circle
    cv2.circle(frame, (x - size // 2, y), size // 2, pink, -1)
    # Right circle
    cv2.circle(frame, (x + size // 2, y), size // 2, pink, -1)

    # Bottom triangle
    pts = np.array([
        [x - size, y],
        [x + size, y],
        [x, y + size]
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], pink)

    # --- STAR SPARKLE ---
    cv2.line(frame, (x + size, y - size), (x + size + 6, y - size), yellow, 1)
    cv2.line(frame, (x + size + 3, y - size - 3), (x + size + 3, y - size + 3), yellow, 1)

def draw_hearts(frame):
    for heart in hearts[:]:
        draw_heart(frame, heart["x"], heart["y"], heart["size"])
        heart["y"] -= heart["speed"]

        if heart["y"] < 0:
            hearts.remove(heart)


cap = cv2.VideoCapture(0)

print("👉 Make ❤️ using THUMB + INDEX finger")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            thumb = hand.landmark[4]
            index = hand.landmark[8]

            h, w, _ = frame.shape
            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            dist = math.hypot(x2 - x1, y2 - y1)

            # DEBUG LINE
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Dist: {int(dist)}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # ❤️ HEART GESTURE
            if dist < 40:
                add_heart(x1, y1)

    draw_hearts(frame)

    cv2.imshow("Pink Heart Gesture ✨", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
