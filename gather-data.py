
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
facecascade1= cv2.CascadeClassifier('Haarcascade/haarcascade_frontalface_default.xml')

facecascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml') #type:ignore


name=input('Enter Your name:')
ret=True

while(ret):
    ret,frame=camera.read()
    if ret == True:
        #Converting the frame to grayscale image.
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        face_coordinates=facecascade.detectMultiScale(gray,1.3,4)
        
        for (a,b,w,h) in face_coordinates:
            faces=frame[b:b+h,a:a+w,:]
            resized_faces=cv2.resize(faces,(50,50))
            
            if i % 10==0 and len(face_data) <10:
                face_data.append(resized_faces)
            cv2.rectangle(frame,(a,b),(a+w,b+h),(255,0,0),2)
        i += 1
        
        cv2.imshow('frames',frame)
        
        if cv2.waitKey(1) == 27 or len(face_data) >=10:
            break
    else:
        print('error')
        break
cv2.destroyAllWindows()
camera.release()     

face_data=np.asarray(face_data)
face_data=face_data.reshape(10,-1)

if 'names.pkl' not in os.listdir('face-recognition/'):
    names=[name]*10
    with open ('face-recognition/names.pkl','wb') as file:
        pickle.dump(names,file)
else:
    
    with open('face-recognition/names.pkl','rb') as file:
        names=pickle.load(file)
        
    names=names + [name]*10
    with open('face-recognition/names.pkl', 'wb') as file:
        pickle.dump(names, file)
        
if 'faces.pkl' not in os.listdir('face-recognition/'):
    with open('face-recognition/faces.pkl', 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open('face-recognition/faces.pkl', 'rb') as w:
        faces = pickle.load(w)

    faces = np.append(faces, face_data, axis=0)
    with open('face-recognition/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)
       
        

