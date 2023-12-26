# Face Recognition Project

## Overview

This Face Recognition project consists of two main components: **gather-data.py** and **recognize.py**. The purpose of the project is to gather facial data, train a K-Nearest Neighbors (KNN) classifier, and then use the trained model for real-time face recognition.

## Components

### gather-data.py

1. **Data Gathering:**
   - Captures live video frames using the OpenCV library.
   - Detects faces in each frame using the Haarcascade Frontal Face Classifier.
   - Resizes and stores the faces in a data matrix.
   - Gathers data for a specified number of frames (e.g., 10 frames per person).

2. **Data Storage:**
   - Saves the gathered facial data matrix into 'faces.pkl'.
   - Manages the names of individuals and stores them in 'names.pkl'.

### recognize.py

1. **Model Training:**
   - Loads the previously gathered facial data and labels from 'faces.pkl' and 'names.pkl'.
   - Trains a KNN classifier using the scikit-learn library.

2. **Real-time Recognition:**
   - Captures live video frames.
   - Detects faces in each frame using the Haarcascade Frontal Face Classifier.
   - Resizes and flattens the detected face for prediction.
   - Utilizes the trained KNN classifier to predict the name associated with each detected face.
   - Displays the real-time video feed with face recognition results.

## How to Use

1. Run **gather-data.py**:
   - Execute this script to gather facial data for training.
   - Enter your name when prompted and follow the instructions.
   - The script will save the gathered data in the 'data/' directory.

2. Run **recognize.py**:
   - Execute this script to load the trained model and perform real-time face recognition.
   - The application will display live video with recognized faces and their corresponding names.

## Dependencies

- OpenCV
- scikit-learn
- NumPy

***
#### Feel free to explore and modify the code as needed. Explore Face Recognition project!
