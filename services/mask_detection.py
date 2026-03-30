import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("models/mask_model.h5")

def predict_mask(face_img):
    face = cv2.resize(face_img, (224, 224))
    face = face / 255.0
    face = np.reshape(face, (1, 224, 224, 3))
    
    pred = model.predict(face)[0]
    
    return "Mask" if pred[0] > 0.5 else "No Mask"