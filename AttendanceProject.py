import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'ImageDataSet'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

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




encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)

        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35), (x2, y2), (0, 255, 0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

            cv2.imshow('Webcam',img)
            cv2.waitKey(1)


# faceLoc = face_recognition.face_locations(imgBuhari)[0]
# encodeBuhari = face_recognition.face_encodings(imgBuhari)[0]
# cv2.rectangle(imgBuhari,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest = face_recognition.face_locations(imgBuhariTest)[0]
# encodeBuhariTest = face_recognition.face_encodings(imgBuhariTest)[0]
# cv2.rectangle(imgBuhariTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results =face_recognition.compare_faces([encodeBuhari],encodeBuhariTest)
# faceDis = face_recognition.face_distance([encodeBuhari],encodeBuhariTest)

#imgBuhari = face_recognition.load_image_file('ImageDataSet/buhari.jpg')
#imgBuhari = cv2.cvtColor(imgBuhari,cv2.COLOR_BGR2RGB)
#imgBuhariTest = face_recognition.load_image_file('ImageDataSet/buharitest.jpg')
#imgBuhariTest = cv2.cvtColor(imgBuhariTest,cv2.COLOR_BGR2RGB)