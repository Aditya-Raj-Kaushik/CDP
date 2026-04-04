import cv2
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: hides the oneDNN log

from utils.face_detector import detect_faces
from services.face_recognition import recognize_face
from services.mask_detection import predict_mask
from services.emotion_detection import predict_emotion

import cv2
from utils.face_detector import detect_faces
from services.face_recognition import recognize_face
from services.mask_detection import predict_mask
from services.emotion_detection import predict_emotion

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    faces = detect_faces(frame)
    
    for face, (x,y,w,h) in faces:
        
        identity = recognize_face(face)
        mask = predict_mask(face)
        emotion = predict_emotion(face)
        
        label = f"{identity} | {mask} | {emotion}"
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    
    cv2.imshow("Smart System", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()