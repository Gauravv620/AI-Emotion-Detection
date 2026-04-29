from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from transformers import pipeline

app = Flask(__name__)
CORS(app)


face_model = tf.keras.models.load_model(
    "backend/models/face_emotion_cnn.keras"
)

face_emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


text_emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)


@app.route("/")
def home():
    return "Face & Text Emotion Detection API (BERT) is running"

# ---------- FACE EMOTION ----------
@app.route("/predict-face", methods=["POST"])
def predict_face():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0

    prediction = face_model.predict(img, verbose=0)
    emotion = face_emotion_labels[int(np.argmax(prediction))]
    confidence = float(np.max(prediction))

    return jsonify({
        "emotion": emotion,
        "confidence": round(confidence * 100, 2)
    })

# ---------- TEXT EMOTION (BERT) ----------
@app.route("/predict-text", methods=["POST"])
def predict_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    outputs = text_emotion_classifier(text)

    # Normalize HuggingFace output
    if isinstance(outputs, list) and len(outputs) > 0:
        results = outputs[0] if isinstance(outputs[0], list) else outputs
    else:
        return jsonify({"error": "Model returned no output"}), 500

    best = max(results, key=lambda x: float(x["score"]))

    return jsonify({
        "emotion": best["label"],
        "confidence": round(float(best["score"]) * 100, 2)
    })




from flask import Flask, render_template, Response
from camera import generate_frames

app = Flask(__name__, template_folder="../frontend/templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video():
    return Response(generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame")



if __name__ == "__main__":
    app.run(debug=True)
