import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Recreate SIMPLE architecture (common mask model)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

model.load_weights(os.path.join("models", "mask_model.h5"))

def predict_mask(face):
    img = cv2.resize(face, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    return "Mask" if pred[0] > pred[1] else "No Mask"