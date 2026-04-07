import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import numpy as np
import cv2
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join("models", "mask_model.h5")
model = load_model(MODEL_PATH)

def predict_mask(face):
    img = cv2.resize(face, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    return "Mask" if pred[0] > pred[1] else "No Mask"