import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ==============================
# Load Models
# ==============================

mask_model = load_model("masked_unmasked_model.h5")
emotion_model = load_model("emotion_model.h5")

# Emotion labels (FER2013 order)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Starting camera... Press 'q' to exit.")

# ==============================
# Live Loop
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        # ==============================
        # 1️⃣ MASK PREDICTION
        # ==============================

        mask_face = cv2.resize(face, (224, 224))
        mask_face = mask_face.astype("float32") / 255.0
        mask_face = np.expand_dims(mask_face, axis=0)

        mask_pred = mask_model.predict(mask_face, verbose=0)[0][0]

        # Assuming:
        # 0 → Mask
        # 1 → No Mask
        if mask_pred < 0.5:
            label = "Mask 😷"
            color = (0, 255, 0)

        else:
            # ==============================
            # 2️⃣ EMOTION PREDICTION
            # ==============================

            emotion_face = cv2.resize(face, (224, 224))
            emotion_face = emotion_face.astype("float32") / 255.0
            emotion_face = np.expand_dims(emotion_face, axis=0)

            emotion_pred = emotion_model.predict(emotion_face, verbose=0)
            emotion_index = np.argmax(emotion_pred)
            emotion_label = emotion_labels[emotion_index]

            label = f"No Mask - {emotion_label}"
            color = (0, 0, 255)

        # Draw Rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Put Text
        cv2.putText(
            frame,
            label,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Mask + Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
