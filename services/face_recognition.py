import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

DATASET_PATH = "data/known"

known_embeddings = []
known_labels = []

# =========================
# LOAD FACES
# =========================
def load_faces():
    print("Loading known faces...")

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (160, 160))
            embedding = embedder.embeddings([img])[0]

            known_embeddings.append(embedding)
            known_labels.append(person)

    print(f"Loaded {len(known_embeddings)} faces")

load_faces()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def recognize_face(face):
    face = cv2.resize(face, (160, 160))
    embedding = embedder.embeddings([face])[0]

    best_score = -1
    identity = "Unknown"

    for known_emb, label in zip(known_embeddings, known_labels):
        score = cosine_similarity(embedding, known_emb)

        if score > best_score:
            best_score = score
            identity = label

    if best_score > 0.55:
        return f"{identity} ({best_score:.2f})"
    else:
        return f"Unknown ({best_score:.2f})"