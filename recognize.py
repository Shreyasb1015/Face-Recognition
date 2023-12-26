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
        
        for (a,b,w,h) in face_coordinates:
            fc=frame[b:b+h,a:a+w,:]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
            text=knn.predict(r)
            cv2.putText(frame,text[0],(a,b-10),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)
            cv2.rectangle(frame,(a,b),(a+w,b+w),(0,0,255),2)
        
        cv2.imshow('Face recognition mode on',frame)
        if cv2.waitKey(1)==27:
            break
    else:
        print("Some error ocurred!!")
        break

cv2.destroyAllWindows()
camera.release()
        
        