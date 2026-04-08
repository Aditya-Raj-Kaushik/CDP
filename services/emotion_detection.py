import cv2
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model


base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None   
)

for layer in base_model.layers[:-50]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.6)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


model.load_weights(os.path.join("models", "emotion_model.h5"))

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

def predict_emotion(face):
    img = cv2.resize(face, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]
    return emotion_labels[np.argmax(pred)]