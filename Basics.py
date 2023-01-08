import cv2
import numpy as np
import face_recognition

imgBuhari = face_recognition.load_image_file('ImageDataSet/buhari.jpg')
imgBuhari = cv2.cvtColor(imgBuhari,cv2.COLOR_BGR2RGB)
imgBuhariTest = face_recognition.load_image_file('ImageDataSet/buharitest.jpg')
imgBuhariTest = cv2.cvtColor(imgBuhariTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgBuhari)[0]
encodeBuhari = face_recognition.face_encodings(imgBuhari)[0]
cv2.rectangle(imgBuhari,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgBuhariTest)[0]
encodeBuhariTest = face_recognition.face_encodings(imgBuhariTest)[0]
cv2.rectangle(imgBuhariTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results =face_recognition.compare_faces([encodeBuhari],encodeBuhariTest)
faceDis = face_recognition.face_distance([encodeBuhari],encodeBuhariTest)
print(results,faceDis)
cv2.putText(imgBuhariTest,f'{results} {round(faceDis[0],2)}', (50,50), cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)


cv2.imshow('buhari', imgBuhari)
cv2.imshow('buharitest', imgBuhariTest)
cv2.waitKey(0)

