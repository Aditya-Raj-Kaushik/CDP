import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

MODEL_PATH = os.path.join("models", "emotion_model.h5")
model = load_model(MODEL_PATH)

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(face):
    img = cv2.resize(face, (224, 224))  # ensure same as training
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    return emotion_labels[np.argmax(pred)]