import os
import cv2
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


from utils.face_detector import detect_faces
from services.mask_detection import predict_mask
from services.emotion_detection import predict_emotion

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = detect_faces(frame)

    for face, (x, y, w, h) in faces:

        # Predictions
        mask = predict_mask(face)
        emotion = predict_emotion(face)

        label = f"{mask} | {emotion}"

        # Color logic
        color = (0, 255, 0) if mask == "Mask" else (0, 0, 255)

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

    cv2.imshow("Smart System (Mask + Emotion)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()