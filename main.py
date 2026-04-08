import os
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.face_detector import detect_faces
from services.face_recognition import recognize_face
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
        identity = recognize_face(face)
        mask = predict_mask(face)
        emotion = predict_emotion(face)

        label = f"{identity} | {mask} | {emotion}"

        # Color logic
        if "Mask" in mask:
            color = (0, 255, 0)
        elif "No Mask" in mask:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Background text box
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, -1)

        cv2.putText(
            frame,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    cv2.imshow("Smart System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()