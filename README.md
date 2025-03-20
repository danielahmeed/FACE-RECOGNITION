# Face Recognition Attendance System

<img src="https://res.cloudinary.com/dagggqd6g/image/upload/f_auto,q_auto/u0zegnahgcc4sedaz4ff"/>

This project implements a face recognition attendance system using OpenCV, face_recognition, and a webcam. It detects and recognizes faces in real-time and marks attendance by storing the recognized faces with timestamps in a CSV file.

## Overview

- The system captures video from the webcam.
- It detects and recognizes faces in the video stream.
- It marks the attendance by writing the recognized face's name, date, and time to a CSV file.

## Prerequisites

- Python 3.x
- OpenCV
- face_recognition
- numpy
- A webcam or an external camera

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/your-username/face-recognition-attendance.git
    cd face-recognition-attendance
    ```

2. Install the required packages:
    ```sh
    pip install opencv-python face-recognition numpy
    ```

3. Prepare your images directory:
    - Place the images of the people you want to recognize inside the `images` directory.
    - Create a subfolder for each person, and place their images inside their respective subfolder.
    - The structure should look like this:
        ```
        images/
        ├── Person1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        ├── Person2/
        │   ├── img1.jpg
        │   ├── img2.jpg
        │   └── ...
        └── ...
        ```

## Usage

1. Run the script:
    ```sh
    python face_recognition_attendance.py
    ```

2. The webcam will start, and the system will begin detecting and recognizing faces.

3. The recognized faces will be displayed with bounding boxes and names.

4. Attendance will be marked in the `Attendance.csv` file with the format: `Name,Date,Time`.

## Troubleshooting

- **No module named 'cv2'**: Make sure OpenCV is installed using `pip install opencv-python`.
- **No module named 'face_recognition'**: Install the face_recognition package using `pip install face-recognition`.
- **FileNotFoundError**: Ensure the directory structure is correctly set up with the `images` directory and subfolders for each person.
- **Webcam not detected**: Check if your webcam is properly connected and accessible. Try using another application to verify.
