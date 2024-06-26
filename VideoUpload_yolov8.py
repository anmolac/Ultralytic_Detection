# -*- coding: utf-8 -*-
"""
Created on Thu May 16 19:27:04 2024

@author: Acharya Anmol
"""



from ultralytics import YOLO 
import cv2


# load yolov8 model
model = YOLO('yolov8n.pt')


# load video
#video_path = './test.mp4'
cap = cv2.VideoCapture(0)



ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:

        # detect objects
        # track objects
        results = model.track(frame, persist=True)#persist make the frame is true

        # plot results
        # cv2.rectangle
        # cv2.putText
        frame_ = results[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break


#results = model(source=0, show=True, conf=0.4, save=True)




#---------------------------------------------------------------------
 