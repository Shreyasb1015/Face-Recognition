import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import numpy as np

facecascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #type:ignore

with open('face-recognition/faces.pkl','rb') as w:
    faces=pickle.load(w)

with open('face-recognition/names.pkl','rb') as file:
    labels=pickle.load(file)

camera=cv2.VideoCapture(0)
