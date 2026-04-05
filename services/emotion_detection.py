import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

model = load_model("models/emotion_model.h5")

EMOTIONS = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def predict_emotion(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48,48))
    face = face / 255.0
    face = np.reshape(face, (1,48,48,1))
    
    pred = model.predict(face)[0]
    return EMOTIONS[np.argmax(pred)]