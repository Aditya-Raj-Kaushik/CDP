import cv2
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# =========================
# REBUILD MODEL (EXACT SAME)
# =========================
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights=None   # IMPORTANT
)

# Match your training freeze
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

output = Dense(1, activation="sigmoid")(x)  # IMPORTANT

model = Model(inputs=base_model.input, outputs=output)

# =========================
# LOAD WEIGHTS
# =========================
model.load_weights(os.path.join("models", "mask_model.h5"))

# =========================
# PREDICTION
# =========================
def predict_mask(face):
    img = cv2.resize(face, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0][0]

    return "Mask" if pred > 0.5 else "No Mask"