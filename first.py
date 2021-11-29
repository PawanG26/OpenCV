# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:43:01 2021

@author: Pawan
"""

import cv2

video = cv2.VideoCapture(0)  

while True:
    ret,frame = video.read()
    
    cv2.imshow("Age-Gender",frame)
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()
   