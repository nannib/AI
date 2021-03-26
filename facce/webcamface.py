# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:51:19 2021

@author: nannib
"""
import cv2
import face_recognition

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#cap = cv2.VideoCapture('20191211_103305.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    resized = cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    rgb_image = resized[:,:,::-1]
    for (top,right,bottom,left) in face_recognition.face_locations(rgb_image):
        cv2.rectangle(resized, (left,top), (right,bottom), (0,0,255))
    cv2.imshow("cap", resized)
    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break
        
cap.release()
cv2.destroyAllWindows()
    