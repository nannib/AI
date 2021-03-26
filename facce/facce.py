# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 09:18:08 2021

@author: nannib
"""
import face_recognition
import cv2

image = face_recognition.load_image_file("folla.jpg")
img = cv2.imread("folla.jpg")
for (top,right,bottom,left) in face_recognition.face_locations(image):
    cv2.rectangle(img, (left,top), (right,bottom), (0,0,255))

cv2.imshow("folla", img)
while True:
    if cv2.waitKey(1) & 0xFF == ord ("q"):
        break
cv2.destroyAllWindows()