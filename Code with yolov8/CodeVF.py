import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO



model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
saisie = 10000000000
saisie2 = 10000000000
saisieYN = 10000000000

while (saisieYN != "y" and saisieYN != "n"):
    print("Press y if you want to see the original video, press n if you don't")
    saisieYN = input("y/n : ")
    
while (saisie != "1" and saisie != "2"):
    print("Please choose between option 1 and option 2")
    saisie = input(" 1 for live video -- 2 for pre recorded video: ")
    if (saisie == "1"):
        cap=cv2.VideoCapture(0)
    if (saisie == "2"):
        while (saisie2 != "1" and saisie2 != "2" and saisie2 != "3" and saisie2 != "4"):
            saisie2 = input("Choose the video : 1 - 2 - 3 - 4: ")
            cap=cv2.VideoCapture("videos/test"+saisie2+".mp4")
    

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
#print(class_list)
count=0


while True:
    
    ret,frame = cap.read()
    if not ret:
        break

    results=model.predict(frame, show=True)
           
    if (saisieYN == "y"):
        cv2.imshow("Original", frame)
        
    if (saisie == "1"):
        frame=cv2.resize(frame,(1020,500))
    
    if (saisie == "2"):
        frame=cv2.resize(frame,(1200,700))
        
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()