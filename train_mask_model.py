import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator# type: ignore
from tensorflow.keras.applications import MobileNetV2# type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout# type: ignore
from tensorflow.keras.models import Model# type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# =========================
# PATH
# =========================
DATASET_PATH = "data/mask_dataset/images"

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# =========================
# DATA AUGMENTATION (STRONG)
# =========================
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.4,
    shear_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.6, 1.4]
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =========================
# MODEL
# =========================
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# 🔥 Fine-tuning (IMPORTANT)
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# =========================
# COMPILE
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# CALLBACKS
# =========================
callbacks = [
    EarlyStopping(
        monitor="loss",
        patience=3,
        restore_best_weights=True
    )
]

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    callbacks=callbacks
)

# =========================
# SAVE
# =========================
os.makedirs("models", exist_ok=True)
model.save("models/mask_model.h5")

print("✅ High-accuracy model saved!")