import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

nltk.download("stopwords")

# ===============================
# LOAD DATA
# ===============================
train_df = pd.read_csv("data/text_emotion/emotion_train.csv")
val_df   = pd.read_csv("data/text_emotion/emotion_val.csv")

print("Train samples:", len(train_df))
print("Train label count:", Counter(train_df.iloc[:, 1]))
print("Val samples:", len(val_df))
print("Val label count:", Counter(val_df.iloc[:, 1]))

# ===============================
# CLEAN TEXT
# ===============================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()
    words = [w for w in text.split() if w.isalpha() and w not in stop_words]
    return " ".join(words)

train_df.iloc[:, 0] = train_df.iloc[:, 0].apply(clean_text)
val_df.iloc[:, 0]   = val_df.iloc[:, 0].apply(clean_text)

# ===============================
# LABEL ENCODING
# ===============================
encoder = LabelEncoder()
y_train = encoder.fit_transform(train_df.iloc[:, 1])
y_val   = encoder.transform(val_df.iloc[:, 1])

print("Emotion classes:", encoder.classes_)

if len(encoder.classes_) < 2:
    raise ValueError("❌ Dataset has only one emotion. Stop.")

# ===============================
# TOKENIZATION
# ===============================
tokenizer = Tokenizer(num_words=8000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df.iloc[:, 0])

X_train = tokenizer.texts_to_sequences(train_df.iloc[:, 0])
X_val   = tokenizer.texts_to_sequences(val_df.iloc[:, 0])

X_train = pad_sequences(X_train, maxlen=100, padding="post")
X_val   = pad_sequences(X_val, maxlen=100, padding="post")

# ===============================
# MODEL
# ===============================
model = Sequential([
    Embedding(8000, 128),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation="relu"),
    Dense(len(encoder.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# TRAIN
# ===============================
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=32
)

# ===============================
# SAVE MODEL
# ===============================
model.save("backend/models/text_emotion_lstm.h5")

with open("backend/models/text_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

with open("backend/models/text_labels.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Text Emotion Model Trained Successfully")
