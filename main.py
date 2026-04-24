from core.attendance import mark_attendance
import cv2
from core.detection import detect_faces
from core.preprocessing import preprocess_face
from core.recognition import compare_faces

cap = cv2.VideoCapture(0)

frame_count = {}

while True:
    ret, frame = cap.read()

    if not ret:
        break

    faces, gray = detect_faces(frame)

    current_names = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = preprocess_face(face)

        name = compare_faces(face)
        current_names.append(name)

        if name != "Unknown":
            frame_count[name] = frame_count.get(name, 0) + 1

            if frame_count[name] == 10:
                mark_attendance(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Reset counts for faces not in frame
    for name in list(frame_count.keys()):
        if name not in current_names:
            frame_count[name] = 0

    cv2.imshow("Recognition System", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()