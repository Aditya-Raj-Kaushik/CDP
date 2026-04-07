import cv2
import numpy as np
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# =========================
# MODEL (FEATURE EXTRACTOR)
# =========================
base_model = MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)

x = base_model.output
x = GlobalAveragePooling2D()(x)

model = Model(inputs=base_model.input, outputs=x)

# =========================
# LOAD KNOWN FACES
# =========================
DATASET_PATH = "data/known"

known_embeddings = []
known_labels = []

def load_known_faces():
    print("🔍 Loading known faces...")

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (224,224))
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            embedding = model.predict(img, verbose=0)[0]

            known_embeddings.append(embedding)
            known_labels.append(person)

    print(f"✅ Loaded {len(known_embeddings)} faces")

load_known_faces()

# =========================
# RECOGNITION FUNCTION
# =========================
def recognize_face(face):
    img = cv2.resize(face, (224,224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    embedding = model.predict(img, verbose=0)[0]

    min_dist = float("inf")
    identity = "Unknown"

    for known_emb, label in zip(known_embeddings, known_labels):
        dist = np.linalg.norm(embedding - known_emb)

        if dist < min_dist:
            min_dist = dist
            identity = label

    if min_dist < 12:
        return f"{identity} ({min_dist:.2f})"
    else:
        return f"Unknown ({min_dist:.2f})"