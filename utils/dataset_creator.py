import cv2
import os
import time

name = input("Enter student name: ")
path = f"dataset/{name}"

if not os.path.exists(path):
    os.makedirs(path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working")
    exit()

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

count = 0

print("Look at camera... Capturing starts in 3 seconds")
time.sleep(3)

while True:
    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        # 🔥 Use COLOR image (not gray)
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (200,200))

        file_name = f"{path}/{count}.jpg"
        cv2.imwrite(file_name, face)

        count += 1

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, f"Images: {count}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        time.sleep(0.3)

    cv2.imshow("Dataset Creator", frame)

    if cv2.waitKey(1) == 27:
        break

    if count >= 50:
        print("High-quality dataset collected!")
        break

cap.release()
cv2.destroyAllWindows()