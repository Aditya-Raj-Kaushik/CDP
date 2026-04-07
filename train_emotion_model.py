import os
import numpy as np
import tensorflow as tf

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

# =========================
# CONFIG
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32

TRAIN_DIR = "data/emotion_dataset/train/"
VAL_DIR = "data/emotion_dataset/test/"

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess(img):
    img = preprocess_input(img)   # EfficientNet-specific normalization
    return img

# =========================
# DATA GENERATORS
# =========================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4]
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
# CLASS WEIGHTS
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
# MODEL
# =========================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Phase 1: Freeze entire base
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

output = Dense(7, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE (PHASE 1)
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        "best_emotion_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
]

# =========================
# TRAIN PHASE 1
# =========================
print("\n🚀 Phase 1: Training top layers...\n")

history1 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# PHASE 2: FINE-TUNING
# =========================
print("\n🔥 Phase 2: Fine-tuning deeper layers...\n")

for layer in base_model.layers[-100:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    class_weight=class_weights,
    callbacks=callbacks
)

# =========================
# SAVE FINAL MODEL
# =========================
model.save("final_emotion_model.h5")

print("\n✅ Training complete!")
print("Saved as: final_emotion_model.h5")