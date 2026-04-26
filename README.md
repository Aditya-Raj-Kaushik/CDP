# Smart AI Vision System

## Overview

Smart AI Vision System is a real-time computer vision project built using Python, OpenCV, TensorFlow, and FaceNet-based facial recognition. The system captures live webcam video, detects faces, and performs multiple AI-powered tasks simultaneously:

* Face Detection
* Face Recognition (Known users such as Aditya, Mouli, etc.)
* Mask Detection
* Emotion Detection

This project demonstrates how multiple deep learning models can be integrated into a single live video pipeline for practical surveillance, security, attendance, and smart monitoring applications.

---

# Features

## Real-Time Face Detection

Detects faces from webcam feed using Haar Cascade Classifier.

## Face Recognition

Recognizes registered users from stored image folders using FaceNet embeddings and cosine similarity.

## Mask Detection

Identifies whether a person is wearing a face mask or not using a custom trained MobileNetV2 model.

## Emotion Detection

Predicts facial emotions such as:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

using a custom trained MobileNetV2 model.

## Live Overlay UI

Displays prediction labels directly on webcam stream with bounding boxes.

---

# Project Structure

```text
CDP/
│── main.py
│── models/
│   ├── mask_model.h5
│   ├── emotion_model.h5
│
│── services/
│   ├── face_recognition.py
│   ├── mask_detection.py
│   ├── emotion_detection.py
│
│── utils/
│   ├── face_detector.py
│
│── data/
│   ├── known/
│       ├── Aditya
```

---

# Technologies Used

* Python 3.10
* OpenCV
* TensorFlow / Keras
* MobileNetV2
* FaceNet (keras-facenet)
* NumPy

---

# How It Works

## Step 1: Face Detection

Each webcam frame is scanned to detect faces.

## Step 2: Face Recognition

Detected face is compared with stored known faces using embeddings.

## Step 3: Mask Detection

Detected face is passed into the trained mask classifier.

## Step 4: Emotion Detection

Detected face is passed into the trained emotion classifier.

## Step 5: Display Results

Results are shown in real-time:

```text
Aditya | Mask | Happy
```

---

# Installation

## Clone Repository

```bash
git clone <your-repository-url>
cd CDP
```

## Create Virtual Environment

```bash
python -m venv venv310
venv310\Scripts\activate
```

## Install Requirements

```bash
pip install opencv-python tensorflow keras-facenet numpy
```

---

# Run Project

```bash
python main.py
```

Press **Q** to quit webcam.

---

# Add New Person for Recognition

Create a new folder inside:

```text
data/known/
```

Example:

```text
data/known/Rahul/
```

Add multiple face images of that person.

Next run will automatically load the new identity.

---

# Model Details

## Mask Detection Model

* Base Model: MobileNetV2
* Transfer Learning
* Binary Classification

Output:

* Mask
* No Mask

## Emotion Detection Model

* Base Model: MobileNetV2
* Transfer Learning
* Multi-Class Classification

Output:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

---

# Future Improvements

* Attendance Management System
* Face Tracking
* Multi-Camera Support
* Database Logging
* Web Dashboard
* Higher FPS Optimization
* Advanced Face Recognition Models

---

# Use Cases

* Smart Surveillance
* Classroom Attendance
* Workplace Monitoring
* Safety Compliance
* Emotion Analytics
* AI Vision Demonstration Projects

---

# Author

Developed by **Aditya Raj Kaushik**

---

# License

This project is for educational and learning purposes.
