import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle 
import numpy as np

facecascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #type:ignore

