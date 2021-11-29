# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 22:55:13 2021

@author: Pawan
"""
import cv2

def FindFace(net,frame,confidence_threshold=0.7):
    DNN_frame=frame.copy()
    print(DNN_frame.shape)
    frameHeight=DNN_frame.shape[0]
    frameWidth=DNN_frame.shape[1]
    blob=cv2.dnn.blobFromImage(DNN_frame,1.0,(227,227),[79,88,114],swapRB=True,crop=False)
    net.setInput(blob)
    results=net.forward()
    faceBoxes=[]
    for i in range(results.shape[2]):
        confidence=results[0,0,i,2]
        if confidence>confidence_threshold:
            x1=int(results[0,0,i,3]*frameWidth)
            y1=int(results[0,0,i,4]*frameHeight)
            x2=int(results[0,0,i,5]*frameWidth)
            y2=int(results[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(DNN_frame,(x1,y1),(x2,y2),(0,255,0),int(round(frameHeight/150)),8)
    return DNN_frame,faceBoxes
        
    
face_Protocol='opencv_face_detector.pbtxt'
faceModel='opencv_face_detector_uint8.pb'
ageProtocol='age_deploy.prototxt'
ageModel='age_net.caffemodel'
gender_Protocol='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'

gender_List=['Male','Female']
age_List=['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']

faceNet=cv2.dnn.readNet(faceModel,face_Protocol)
ageNet=cv2.dnn.readNet(ageModel,ageProtocol)
genderNet=cv2.dnn.readNet(genderModel,gender_Protocol)

video=cv2.VideoCapture(0)
padding=20
while cv2.waitKey(1)<0:
    hasFrame,frame=video.read()
    if not hasFrame:
        cv2.waitKey()
        break
        
    resultImg,faceBoxes=FindFace(faceNet,frame)
    
    if not faceBoxes:
        print("No face detected")
    
    for i in faceBoxes:
        face=frame[max(0,i[1]-padding):min(i[3]+padding,frame.shape[0]-1),max(0,i[0]-padding):min(i[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227),[79,88,114],swapRB=True,crop=False)
        genderNet.setInput(blob)
        genPrediction=genderNet.forward()
        gender=gender_List[genPrediction[0].argmax()]
        
        ageNet.setInput(blob)
        agePrediction=ageNet.forward()
        age=age_List[agePrediction[0].argmax()]
        cv2.putText(resultImg,f'{gender},{age}',(i[0],i[1]-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow("Detecting Age / Gender",resultImg)
        
        
        if cv2.waitKey(30) & 0xFF == ord('b'):
            break
            
cv2.destroyAllWindows()
        