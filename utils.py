import cv2
import numpy as np

def empty(val):
    pass

def getContours(img, cannyThr=[200,200], showCanny=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5),1)
    imgCanny = cv2.Canny(imgBlur, cannyThr[0], cannyThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny: cv2.imshow("canny", imgCanny)

