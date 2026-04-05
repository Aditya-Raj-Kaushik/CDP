import os
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/test"

# =========================
# FACE DETECTION
# =========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        img = img[y:y+h, x:x+w]

    return img


# =========================
# PREPROCESS FUNCTION (FIXED)
# =========================
def preprocess(img):
    img = crop_face(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 🔥 FIX: dtype + normalization
    img = img.astype("float32") / 255.0

    return img


# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("Classes:", train_generator.class_indices)


# =========================
# CLASS WEIGHTS (BALANCE DATA)
# =========================
labels = train_generator.classes
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


# =========================
# MODEL (TRANSFER LEARNING)
# =========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze most layers initially
for layer in base_model.layers[:-20]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)

output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)


# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# CALLBACKS
# =========================
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)


# =========================
# TRAIN
# =========================
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[reduce_lr, early_stop]
)


# =========================
# SAVE MODEL (.h5)
# =========================
model.save("emotion_model.h5")

print("✅ Emotion model saved as emotion_model.h5")