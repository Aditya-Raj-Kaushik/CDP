import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_list = []
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_list.append((face, (x, y, w, h)))
    
    return face_list