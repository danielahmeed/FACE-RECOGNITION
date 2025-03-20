import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to images
path = "images"
images = []
classNames = []

# List all subfolders (each folder represents a person)
subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
print("Total students detected:", len(subfolders))

# Load images and corresponding names
for subfolder in subfolders:
    personName = os.path.basename(subfolder)
    imageFiles = os.listdir(subfolder)
    for imageFile in imageFiles:
        curImg = cv2.imread(os.path.join(subfolder, imageFile))
        if curImg is not None:
            images.append(curImg)
            classNames.append(personName)

# Function to find encodings of known images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:  # Ensure at least one face is detected
            encodeList.append(encodings[0])
        else:
            print("Warning: No face detected in one of the training images!")
    return encodeList

# Function to mark attendance
def markAttendance(name):
    filename = "Attendance.csv"
    file_exists = os.path.isfile(filename)

    with open(filename, "a") as f:  # Open in append mode
        if not file_exists:
            f.write("Name,Date,Time\n")  # Write header if file doesn't exist

        now = datetime.now()
        dateString = now.strftime("%Y-%m-%d")
        timeString = now.strftime("%H:%M:%S")

        # Check if the name is already recorded for today
        with open(filename, "r") as fr:
            lines = fr.readlines()
            for line in lines:
                if name in line and dateString in line:
                    return  # Skip duplicate entry for the same day

        # Write new attendance entry
        f.write(f"{name},{dateString},{timeString}\n")

# Encode known images
encodeListKnown = findEncodings(images)
print("Encoding Complete. Total Encoded Faces:", len(encodeListKnown))

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use CAP_DSHOW for better compatibility on Windows

if not cap.isOpened():
    print("Error: Cannot access webcam.")
    exit()

while True:
    success, img = cap.read()
    
    if not success or img is None:
        print("Warning: Failed to capture frame. Retrying...")
        continue  # Skip the loop iteration if no frame is captured

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # Resize for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        
        if len(matches) > 0 and len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex] and faceDis[matchIndex] < 0.6:  # Adjustable threshold
                name = classNames[matchIndex].upper()
            else:
                name = "UNKNOWN"
        else:
            name = "UNKNOWN"

        # Draw rectangle & label
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back to original size
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        # Mark attendance if known face is detected
        if name != "UNKNOWN":
            markAttendance(name)

    cv2.imshow("Webcam", img)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
