import cv2
import numpy as np
import utils

webcam = False
path = 'resources/screws1.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920) # Video Dimensions
cap.set(4,1080)
scale = 3
widthPaper = 279 * scale #21.59cm
heightPaper = 216 * scale #27.94cm

cv2.namedWindow("TrackBars") #Names need to stay the same
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Background Threshold 1", "TrackBars", 255,255,utils.empty)
cv2.createTrackbar("Background Threshold 2", "TrackBars", 60,255,utils.empty)
cv2.createTrackbar("Screw Threshold 1", "TrackBars", 197,255,utils.empty)
cv2.createTrackbar("Screw Threshold 2", "TrackBars", 68,255,utils.empty)
cv2.createTrackbar("Background Blur", "TrackBars", 5,10,utils.empty)
cv2.createTrackbar("Screw Blur", "TrackBars", 7,10,utils.empty)

while True:
    if webcam: success, img = cap.read()
    else: img = cv2.imread(path)

    # contours bigger than 1000, with 6 edges!?
    thrBg1 = cv2.getTrackbarPos("Background Threshold 1", "TrackBars") #255
    thrBg2 = cv2.getTrackbarPos("Background Threshold 2", "TrackBars") #60
    thrS1 = cv2.getTrackbarPos("Screw Threshold 1", "TrackBars")
    thrS2 = cv2.getTrackbarPos("Screw Threshold 2", "TrackBars")
    blurBg = cv2.getTrackbarPos("Background Blur", "TrackBars") #255
    blurS = cv2.getTrackbarPos("Screw Blur", "TrackBars")
    if blurBg % 2 == 0:
        blurBg += 1
    if blurS % 2 == 0:
        blurS += 1
    img, contours = utils.getContours(img, cannyThr=[thrBg1, thrBg2], gaussianBlur=(blurBg, blurBg), showCanny=False, minArea=50000, filterEdges=4, draw=False)

    if len(contours) != 0:
        biggest = contours[0][2]
        print(biggest)
        imgWarp = utils.warpImg(img, biggest, widthPaper, heightPaper)
        cv2.imshow("A4", imgWarp)
        img2, contours2 = utils.getContours(imgWarp, cannyThr=[thrS1, thrS2],
                                            gaussianBlur=(blurS, blurS), showCanny=True,
                                            minArea=0, filterEdges=0, draw=True)
        cv2.imshow("Image 2", img2)


    else:
        print("NO CONTOURS")

    img = cv2.resize(img, (0,0),None,1.5,1.5) # Scales after weve manipulated the photo
    cv2.imshow("Original", img)
    cv2.waitKey(1)