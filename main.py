import cv2
import numpy as np
import utils

webcam = False
path = 'resources/screws6.jpg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920) # Video Dimensions
cap.set(4,1080)
scale = 100
widthPaper = int(11 * scale) #27.94cm = 11"
heightPaper = int(8.5 * scale) #21.59cm = 8.5"

cv2.namedWindow("TrackBars") #Names need to stay the same
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Background Threshold 1", "TrackBars", 255,255,utils.empty)
cv2.createTrackbar("Background Threshold 2", "TrackBars", 60,255,utils.empty)
cv2.createTrackbar("Screw Threshold 1", "TrackBars", 65,255,utils.empty)
cv2.createTrackbar("Screw Threshold 2", "TrackBars", 52,255,utils.empty)
cv2.createTrackbar("Background Blur", "TrackBars", 5,10,utils.empty)
cv2.createTrackbar("Screw Blur", "TrackBars", 7,10,utils.empty)
# TODO: Add min and max area Trackbars for screws

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
        # cv2.imshow("A4", imgWarp)
        # TODO: Could add maxArea similar to minArea so we don't detect large objects (multiple screws on top of each other)
        #       Might also be nice to seperately detect the larger object as unidentified to seperate them
        img2, contours2 = utils.getContours(imgWarp, cannyThr=[thrS1, thrS2],
                                            gaussianBlur=(blurS, blurS), showCanny=True,
                                            minArea=0, filterEdges=0, draw=True)
        if len(contours2) != 0:
            for contour in contours2:
                rect = cv2.minAreaRect(contour[2])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                im = cv2.drawContours(img2, [box], 0, (0, 0, 255), 2)
                nPoints = utils.reorder(box)
                screw_w = round(utils.findDist(nPoints[0], nPoints[1])/scale, 3)
                screw_h = round(utils.findDist(nPoints[0], nPoints[2])/scale, 3)
                # x, y, w, h = cv2.boundingRect(contour[2]) #Straight rectangle
                # cv2.rectangle(img2, (x,y), (x+w, y+h), (0,255, 0), 2)
                # cv2.polylines(img2,[contour[2]], True, (0,255,0), 2) #True for closed
                x, y, w, h = contour[3]
                cv2.putText(img2, '{} in'.format(screw_w), (x, y+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
                cv2.putText(img2, '{} in'.format(screw_h), (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 1)
        cv2.imshow("Image 2", img2)
    else:
        print("NO CONTOURS")

    img = cv2.resize(img, (0,0),None,1.5,1.5) # Scales after weve manipulated the photo
    cv2.imshow("Original", img)
    cv2.waitKey(1)