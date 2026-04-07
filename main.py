import os
import cv2

os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from utils.face_detector import detect_faces
from services.mask_detection import predict_mask
from services.emotion_detection import predict_emotion
from services.face_recognition import recognize_face   # ✅ NEW

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for face, (x, y, w, h) in faces:

        # =========================
        # PREDICTIONS
        # =========================
        identity = recognize_face(face)      # ✅ NEW
        mask = predict_mask(face)
        emotion = predict_emotion(face)

        label = f"{identity} | {mask} | {emotion}"

        # =========================
        # COLOR LOGIC (IMPROVED)
        # =========================
        if "Mask" in mask:
            color = (0, 255, 0)       # Green
        elif "No Mask" in mask:
            color = (0, 0, 255)       # Red
        else:
            color = (0, 255, 255)     # Yellow (Uncertain)

        # =========================
        # DRAW BOX
        # =========================
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Background for text (better visibility)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)

        # Text
        cv2.putText(
            frame,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),   # black text
            1
        )

    cv2.imshow("Smart System (Face + Mask + Emotion)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()