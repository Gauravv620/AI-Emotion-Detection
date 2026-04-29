import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import math
import random


model = tf.keras.models.load_model(
    "backend/models/face_emotion_cnn.keras"
)

emotion_labels = [
    "Angry", "Disgust", "Fear",
    "Happy", "Sad", "Surprise", "Neutral"
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)


hearts = []

def add_heart(x, y):
    hearts.append({"x": x, "y": y, "size": 18})

def draw_heart(frame, x, y, size):
    x, y, size = int(x), int(y), int(size)
    pink = (203, 192, 255)

    cv2.circle(frame, (x-size//2, y), size//2, pink, -1)
    cv2.circle(frame, (x+size//2, y), size//2, pink, -1)

    pts = np.array([[x-size, y], [x+size, y], [x, y+size]], np.int32)
    cv2.fillPoly(frame, [pts], pink)

cap = cv2.VideoCapture(0)

def generate_frames():
    global hearts

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w, _ = frame.shape

        # ---------- FACE EMOTION ----------
        is_happy = False
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, fw, fh) in faces:
            roi = gray[y:y+fh, x:x+fw]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.reshape(1, 48, 48, 1) / 255.0

            preds = model.predict(roi, verbose=0)
            idx = np.argmax(preds)
            emotion = emotion_labels[idx]
            conf = np.max(preds) * 100

            if emotion == "Happy" and conf >= 40:
                is_happy = True

            cv2.rectangle(frame, (x,y), (x+fw,y+fh), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} ({conf:.1f}%)",
                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0,255,0), 2)
            break

        # ---------- HAND GESTURE ----------
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                thumb = hand.landmark[4]
                index = hand.landmark[8]

                x1, y1 = int(thumb.x*w), int(thumb.y*h)
                x2, y2 = int(index.x*w), int(index.y*h)
                dist = math.hypot(x2-x1, y2-y1)

                if is_happy and dist < 40:
                    add_heart(x1, y1)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # ---------- DRAW HEARTS ----------
        for heart in hearts[:]:
            heart["y"] -= 3
            draw_heart(frame, heart["x"], heart["y"], heart["size"])
            if heart["y"] < 0:
                hearts.remove(heart)

        # ---------- STREAM ----------
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
