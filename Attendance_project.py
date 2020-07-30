import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'
images=[]
classNames=[]
myList=os.listdir(path)
print(myList)

for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

#function for encoding the Images
def findEncodings(images):
    encodeList=[]
    for img in images:
        #converting the images to RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #encoding the RGB images
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#to check the function is working or not!
encodeListKnown=findEncodings(images)
#print(len(encodeListKnown))
print("Encoding Complete")

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


cap=cv2.VideoCapture(0) #webcam

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)  #small image
    imgS=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facesCurFrame=face_recognition.face_locations(imgS)
    encodesCurFrame=face_recognition.face_encodings(imgS,facesCurFrame)

    #match the current face encodings to our known faces encoding list to find the matches
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #minimum from all the Distances
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                #print(name)
                y1,x2,y2,x1 = faceLoc
                #reviving the orginal img by multiplying by 4
                y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
                markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)
