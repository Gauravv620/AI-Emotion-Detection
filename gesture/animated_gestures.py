import cv2
import mediapipe as mp
import math
import random

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)


hearts = []
likes = []
stars = []
sparkles = []

def add_particle(lst, x, y, color, size=12):
    lst.append({
        "x": int(x),
        "y": int(y),
        "vy": random.randint(1, 4),
        "size": size,
        "color": color
    })

def draw_particles(frame, lst, shape="circle"):
    for p in lst[:]:
        if shape == "circle":
            cv2.circle(frame, (p["x"], p["y"]), p["size"], p["color"], -1)
        elif shape == "star":
            cv2.drawMarker(frame, (p["x"], p["y"]), p["color"],
                           cv2.MARKER_STAR, p["size"]*2, 2)

        p["y"] -= p["vy"]
        if p["y"] < 0:
            lst.remove(p)

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

print("❤️ Heart | 👍 Like | ✋ Sparkle | ✌️ Stars")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        def pt(i):
            return int(lm[i].x * w), int(lm[i].y * h)

        index_up  = finger_up(lm, 8, 6)
        middle_up = finger_up(lm, 12, 10)
        ring_up   = finger_up(lm, 16, 14)
        pinky_up  = finger_up(lm, 20, 18)

        tx, ty = pt(4)
        ix, iy = pt(8)
        dist = math.hypot(tx - ix, ty - iy)

        cx, cy = pt(9)  # palm center

        # ❤️ HEART / PINCH
        if dist < 30:
            for _ in range(2):
                add_particle(hearts, cx, cy, (255, 105, 180), 10)
            cv2.putText(frame, "HEART", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 105, 180), 2)

        # 👍 THUMBS UP
        elif not index_up and not middle_up and not ring_up and not pinky_up:
            add_particle(likes, cx, cy, (0, 255, 0), 12)
            cv2.putText(frame, "LIKE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ✋ OPEN PALM
        elif index_up and middle_up and ring_up and pinky_up:
            for _ in range(3):
                add_particle(sparkles, cx, cy, (0, 255, 255), 6)
            cv2.putText(frame, "SPARKLE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # ✌️ VICTORY
        elif index_up and middle_up and not ring_up and not pinky_up:
            for _ in range(2):
                add_particle(stars, cx, cy, (255, 255, 0), 14)
            cv2.putText(frame, "STARS", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # Draw animations
    draw_particles(frame, hearts, "circle")
    draw_particles(frame, likes, "circle")
    draw_particles(frame, sparkles, "circle")
    draw_particles(frame, stars, "star")

    cv2.imshow("Animated Gesture Effects", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
