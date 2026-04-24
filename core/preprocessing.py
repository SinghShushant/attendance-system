import cv2

def preprocess_face(face):
    face = cv2.resize(face, (200,200))
    face = cv2.GaussianBlur(face, (5,5), 0)
    return face