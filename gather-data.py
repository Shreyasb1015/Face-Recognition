
# Getting all libraries
import os
import numpy as np
import cv2
import pickle

#Creating a list to store the frames of faces.
face_data=[]  
i=0

camera=cv2.VideoCapture(0)

#Loading the haarcascade
facecascade = cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')

name=input('Enter Your name:')
ret=True

while(ret):
    ret,frame=camera.read()
    if ret == True:
        #Converting the frame to grayscale image.
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        

