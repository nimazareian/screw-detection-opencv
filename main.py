import cv2
import numpy as np
import utils

webcam = False
path = 'resources/screws1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920) # Video Dimensions
cap.set(4,1080)

cv2.namedWindow("TrackBars") #Names need to stay the same
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Threshold 1", "TrackBars", 179,179,utils.empty) #In open cv the highest is 179
cv2.createTrackbar("Threshold 2", "TrackBars", 63,179,utils.empty)

while True:
    if webcam: success, img = cap.read()
    else: img = cv2.imread(path)

    thr1 = cv2.getTrackbarPos("Threshold 1","TrackBars")
    thr2 = cv2.getTrackbarPos("Threshold 2","TrackBars")
    print(thr1,thr2)
    utils.getContours(img, cannyThr=[thr1,thr2], showCanny=True)

    img = cv2.resize(img, (0,0),None,1.5,1.5) # Scales after weve manipulated the photo
    cv2.imshow("Original", img)
    cv2.waitKey(1)