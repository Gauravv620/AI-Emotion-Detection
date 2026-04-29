import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
import random

# ===============================
# LOAD FACE EMOTION MODEL
# ===============================
model = tf.keras.models.load_model(
    "backend/models/face_emotion_cnn.keras"
)

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# ===============================
# FACE DETECTOR
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ===============================
# MEDIAPIPE HANDS
# ===============================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===============================
# HEART PARTICLES
# ===============================
hearts = []

def add_heart(x, y):
    hearts.append({
        "x": int(x),
        "y": int(y),
        "size": random.randint(14, 22),
        "speed": random.uniform(1.5, 3.0)
    })

def draw_heart(frame, x, y, size):
    x, y, size = int(x), int(y), int(size)
    pink = (203, 192, 255)  # Pink (BGR)

    # top circles
    cv2.circle(frame, (x - size//2, y), size//2, pink, -1)
    cv2.circle(frame, (x + size//2, y), size//2, pink, -1)

    # bottom triangle
    pts = np.array([
        [x - size, y],
        [x + size, y],
        [x, y + size]
    ], dtype=np.int32)

    cv2.fillPoly(frame, [pts], pink)

def draw_hearts(frame):
    for heart in hearts[:]:
        heart["y"] -= heart["speed"]
        draw_heart(frame, heart["x"], heart["y"], heart["size"])
        if heart["y"] < 0:
            hearts.remove(heart)

# ===============================
# CAMERA
# ===============================
cap = cv2.VideoCapture(0)
print("🙂 Happy + 👍 Index&Thumb = ❤️ Hearts")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape

    # ===============================
    # FACE EMOTION DETECTION
    # ===============================
    emotion = "Unknown"
    confidence = 0.0
    is_happy = False

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, fw, fh) in faces:
        face = gray[y:y+fh, x:x+fw]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0

        preds = model.predict(face, verbose=0)
        idx = int(np.argmax(preds))
        emotion = emotion_labels[idx]
        confidence = float(np.max(preds)) * 100

        if emotion == "Happy" and confidence >= 30:
            is_happy = True

        color = (0, 255, 0) if is_happy else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+fw, y+fh), color, 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )
        break  # only first face

    # ===============================
    # HAND GESTURE (THUMB + INDEX)
    # ===============================
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            thumb = hand.landmark[4]
            index = hand.landmark[8]

            x1, y1 = int(thumb.x * w), int(thumb.y * h)
            x2, y2 = int(index.x * w), int(index.y * h)

            dist = math.hypot(x2 - x1, y2 - y1)

            # ❤️ HEART ONLY IF HAPPY + GESTURE
            if is_happy and dist < 40:
                add_heart(x1, y1)

    draw_hearts(frame)

    cv2.imshow("Emotion + Gesture Interaction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
