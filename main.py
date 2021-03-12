import cv2
import numpy as np
import utils

WEBCAM = False
PATH = 'resources/screws6.jpg'
cap = cv2.VideoCapture(0)
cap.set(10, 160)
cap.set(3, 1920)  # Video Dimensions
cap.set(4, 1080)
scale = 100
widthPaper = int(11 * scale)   # 27.94cm = 11"
heightPaper = int(8.5 * scale) # 21.59cm = 8.5"

cv2.namedWindow("TrackBars")  # Trackbars on same window need to have same name
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Background Threshold 1", "TrackBars", 255, 255, utils.empty)
cv2.createTrackbar("Background Threshold 2", "TrackBars", 60, 255, utils.empty)
cv2.createTrackbar("Screw Threshold 1", "TrackBars", 65, 255, utils.empty)
cv2.createTrackbar("Screw Threshold 2", "TrackBars", 52, 255, utils.empty)
cv2.createTrackbar("min_area_s", "TrackBars", 700, 10000, utils.empty)
cv2.createTrackbar("max_area_s", "TrackBars", 10000, 20000, utils.empty)
cv2.createTrackbar("Background Blur", "TrackBars", 5, 10, utils.empty)
cv2.createTrackbar("Screw Blur", "TrackBars", 7, 10, utils.empty)

while True:
    # Use a picture or webcam
    if WEBCAM:
        success, img = cap.read()
    else:
        img = cv2.imread(PATH)

    # Get trackbar values
    thrBg1 = cv2.getTrackbarPos("Background Threshold 1", "TrackBars")  # 255
    thrBg2 = cv2.getTrackbarPos("Background Threshold 2", "TrackBars")  # 60
    thrS1 = cv2.getTrackbarPos("Screw Threshold 1", "TrackBars")
    thrS2 = cv2.getTrackbarPos("Screw Threshold 2", "TrackBars")
    minArea = cv2.getTrackbarPos("min_area_s", "TrackBars")
    maxArea = cv2.getTrackbarPos("max_area_s", "TrackBars")
    blurBg = cv2.getTrackbarPos("Background Blur", "TrackBars") # 255
    blurS = cv2.getTrackbarPos("Screw Blur", "TrackBars")
    if blurBg % 2 == 0:
        blurBg += 1
    if blurS % 2 == 0:
        blurS += 1

    # Get contours of the background
    img, contours = utils.getContours(img, cannyThr=[thrBg1, thrBg2], gaussianBlur=(blurBg, blurBg),
                                      minArea=50000, filterEdges=4, showCanny=False, drawContours=True)

    # if any contours exist
    if len(contours) != 0:
        # Warp background to fit screen
        biggest = contours[0][2]
        imgWarp = utils.warpImg(img, biggest, widthPaper, heightPaper)
        img2, contours2 = utils.getContours(imgWarp, cannyThr=[thrS1, thrS2],
                                            gaussianBlur=(blurS, blurS), showCanny=True,
                                            minArea=minArea, maxArea=maxArea, filterEdges=0, drawContours=False)
        # Draw box around contour
        if len(contours2) != 0:
            for contour in contours2:
                utils.drawBoxAroundContour(img2, contour, scale=scale, offsetDimension=-(3/16)-0.03, drawPolylines=False)

        cv2.imshow("Image 2", img2)

    img = cv2.resize(img, (0,0),None,1,1) # Scales output photo
    cv2.imshow("Original", img)
    cv2.waitKey(1)