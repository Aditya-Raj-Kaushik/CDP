from deepface import DeepFace

DB_PATH = "data/known_faces"

def recognize_face(face_img):
    try:
        result = DeepFace.find(
            img_path=face_img,
            db_path=DB_PATH,
            enforce_detection=False
        )

        if len(result[0]) > 0:
            return result[0]['identity'][0].split("/")[-2]
        else:
            return "Unknown"

    except Exception as e:
        return "Unknown"