import cv2
import numpy as np
import os

def get_histogram(face):
    # 🔥 Color histogram (3 channels)
    hist = cv2.calcHist([face],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
    return cv2.normalize(hist, hist).flatten()

def get_edges(face):
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 100, 200)

def compare_faces(face, dataset_path="dataset"):
    best_score = -1
    best_match = "Unknown"

    face_hist = get_histogram(face)
    face_edges = get_edges(face)

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)

            img = cv2.imread(img_path)  # 🔥 color image
            img = cv2.resize(img, (200,200))

            hist = get_histogram(img)
            edges = get_edges(img)

            hist_score = cv2.compareHist(face_hist, hist, cv2.HISTCMP_CORREL)
            edge_score = cv2.matchTemplate(face_edges, edges, cv2.TM_CCOEFF_NORMED)[0][0]

            final_score = 0.7 * hist_score + 0.3 * edge_score

            if final_score > best_score:
                best_score = final_score
                best_match = person

    print("Score:", best_score, "Match:", best_match)

    if best_score < 0.6:
        return "Unknown"

    return best_match