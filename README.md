# AI Emotion Detection

An AI-powered emotion detection system that combines **facial expression recognition**, **text emotion classification**, and **hand-gesture interaction** in a single Python-based application.

The project uses a custom **CNN model** for facial emotion recognition, a pretrained **DistilRoBERTa emotion classifier** for text, and **MediaPipe Hands + OpenCV** for real-time hand tracking.

## Features

- Facial emotion detection from images/webcam frames
- Real-time face detection using OpenCV Haar Cascade
- Seven facial emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Text-based emotion detection using `j-hartmann/emotion-english-distilroberta-base`
- Confidence score for predictions
- Real-time webcam emotion detection
- Hand landmark detection using MediaPipe
- Heart animation triggered when a happy expression is detected and a hand gesture is performed
- Separate training scripts for the facial CNN and an LSTM-based text model

## Project Architecture

```text
                    AI Emotion Detection
                            |
          +-----------------+-----------------+
          |                 |                 |
       Face Input        Text Input       Webcam Input
          |                 |                 |
      OpenCV Face       DistilRoBERTa      OpenCV +
      Detection         Emotion Model      MediaPipe
          |                 |                 |
       CNN Model        Emotion +         Face Emotion +
          |             Confidence        Hand Gesture
          |                                   |
    Emotion + Confidence                 Heart Animation
```

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core development language |
| TensorFlow / Keras | CNN model and deep learning |
| OpenCV | Image processing, face detection and webcam streaming |
| MediaPipe | Hand landmark and gesture detection |
| Hugging Face Transformers | Text emotion classification |
| DistilRoBERTa | Pretrained text emotion model |
| Flask | Backend web application/API |
| HTML | Frontend interface |
| NumPy | Numerical processing |
| NLTK | Text preprocessing for the LSTM training pipeline |
| Pandas | Dataset handling |
| Scikit-learn | Label encoding for text-model training |

## Machine Learning Models

### 1. Facial Emotion CNN

The facial emotion model is a convolutional neural network trained on 48x48 grayscale facial images. The architecture contains convolution, max-pooling, dense and dropout layers and produces seven emotion classes.

Training configuration includes:

- Image size: `48 x 48`
- Color mode: grayscale
- Batch size: `64`
- Epochs: `10`
- Optimizer: Adam
- Loss: Categorical Cross-Entropy
- Output activation: Softmax

The trained model is stored in:

```text
backend/models/face_emotion_cnn.keras
```

### 2. Text Emotion Classification

The current backend uses the pretrained Hugging Face model:

```text
j-hartmann/emotion-english-distilroberta-base
```

The model analyzes input text and returns the emotion with the highest confidence score.

### 3. LSTM Text Training Pipeline

The repository also contains `training/train_text_lstm.py`, which provides an alternative LSTM-based text classification pipeline using tokenization, padding, an embedding layer and an LSTM layer.

This script is useful for training and experimentation with a custom text emotion model.

## Dataset Structure

The project expects the following dataset structure:

```text
data/
├── fer2013/
│   └── train/
│       ├── Angry/
│       ├── Disgust/
│       ├── Fear/
│       ├── Happy/
│       ├── Sad/
│       ├── Surprise/
│       └── Neutral/
│
└── text_emotion/
    ├── emotion_train.csv
    └── emotion_val.csv
```

The repository contains dataset directories, while datasets may need to be supplied locally depending on how the project is being run.

## Project Structure

```text
AI-Emotion-Detection/
│
├── backend/
│   ├── app.py
│   ├── camera.py
│   ├── webcam_emotion.py
│   ├── haarcascade_frontalface_default.xml
│   └── models/
│       ├── face_emotion_cnn.h5
│       └── face_emotion_cnn.keras
│
├── data/
│   ├── fer2013/
│   └── text_emotion/
│
├── frontend/
│   └── templates/
│       └── index.html
│
├── gesture/
│
├── training/
│   ├── check_dataset.py
│   ├── train_face_cnn.py
│   └── train_text_lstm.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Gauravv620/AI-Emotion-Detection.git
cd AI-Emotion-Detection
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The application code also uses Flask, Flask-CORS, Transformers, MediaPipe, Pandas, Scikit-learn and NLTK. If these packages are not already present in your environment, install them with:

```bash
pip install flask flask-cors transformers mediapipe pandas scikit-learn nltk
```

## Run the Application

From the project root:

```bash
python backend/app.py
```

The Flask application runs locally and provides the webcam-based emotion and gesture interface.

Open the local address shown by Flask in your browser.

## API Endpoints

The backend contains prediction routes for both supported input types.

### Face Emotion

```text
POST /predict-face
```

Send an image using the form field:

```text
image
```

Example response:

```json
{
  "emotion": "Happy",
  "confidence": 94.21
}
```

### Text Emotion

```text
POST /predict-text
```

Request body:

```json
{
  "text": "I am very happy today!"
}
```

Example response:

```json
{
  "emotion": "joy",
  "confidence": 98.12
}
```

## Real-Time Webcam Pipeline

The webcam component follows this process:

```text
Webcam Frame
     |
     v
Face Detection
     |
     v
48x48 Grayscale ROI
     |
     v
CNN Emotion Prediction
     |
     +----> Happy + confidence >= 40%
     |                 |
     |                 v
     |          Check Hand Gesture
     |                 |
     |                 v
     |          Trigger Heart Animation
     |
     v
Display Emotion + Confidence
```

Hand landmarks are detected with MediaPipe. When a happy facial expression is detected and the thumb/index-finger distance satisfies the gesture condition, a heart animation is added to the webcam stream.

## Training the Face Emotion Model

Make sure the FER2013 training data is available under:

```text
data/fer2013/train/
```

Then run:

```bash
python training/train_face_cnn.py
```

The trained model is saved as:

```text
backend/models/face_emotion_cnn.keras
```

## Training the LSTM Text Model

Place the text datasets at:

```text
data/text_emotion/emotion_train.csv
data/text_emotion/emotion_val.csv
```

Then run:

```bash
python training/train_text_lstm.py
```

The script creates:

```text
backend/models/text_emotion_lstm.h5
backend/models/text_tokenizer.pkl
backend/models/text_labels.pkl
```

## Use Cases

- Human-computer interaction
- Emotion-aware interfaces
- Educational AI demonstrations
- AI/ML academic projects
- Computer vision learning
- Natural language processing experiments
- Real-time webcam emotion analysis
- Multimodal emotion detection research

## Limitations

- Facial emotion predictions can be affected by lighting, camera quality, face angle and occlusion.
- Emotion classification is probabilistic and should not be treated as a definitive measurement of a person's internal emotional state.
- The text classifier is primarily intended for English text.
- Real-time webcam performance depends on available CPU/GPU resources.
- Gesture detection can be affected by hand position, camera angle and visibility.

## Future Improvements

- Add more robust face detection and tracking
- Improve CNN accuracy using data augmentation and a deeper architecture
- Add multi-face emotion tracking
- Add emotion history and analytics
- Add voice/speech emotion recognition
- Combine face, text and gesture predictions into a multimodal emotion score
- Improve the frontend UI and add live prediction dashboards
- Deploy the application using a production WSGI server

## Author

**Gaurav Kumar**

GitHub: [Gauravv620](https://github.com/Gauravv620)

## License

This project is intended for educational and research purposes. Add an appropriate open-source license to the repository if you plan to distribute or reuse the project publicly.
