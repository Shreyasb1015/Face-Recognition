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
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(faces,labels)

while True:
    ret,frame=camera.read()
    if ret==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        face_coordinates=facecascade.detectMultiScale(gray,1.3,5)
        