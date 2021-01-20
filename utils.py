import cv2
import numpy as np

def empty(val):
    pass

def getContours(img, cannyThr=[200,200], gaussianBlur=(5,5), showCanny=False, minArea=1000, filterEdges = 0, draw=False):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, gaussianBlur,1) # Could tune blue value for better edges - GaussianBlue must be ODD and Positive
    imgCanny = cv2.Canny(imgBlur, cannyThr[0], cannyThr[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny: cv2.imshow("canny", imgCanny)

    contours, hiearchy = cv2.findContours(imgThre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i,True) # True for contour being closed
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) # Approximation of edges of contour
            bbox = cv2.boundingRect(approx) # Bounding box
            print("bbox:", bbox)
            if filterEdges > 0: #0  how many edges
                if len(approx) == filterEdges:
                    print("# of edges", len(approx))
                    finalContours.append([len(approx), area, approx, bbox, i]) # Keep contour
            else:
                # print("NOOOT FilterEdges # of edges")
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key = lambda x: x[1], reverse=True) #Sorting based on area (largest first)
    if draw:
        for contour in finalContours:
            cv2.drawContours(img, contour[4], -1, (0,0,255), 3)

    return img, finalContours

def reorder(points):
    print(points.shape)
    pointsNew = np.zeros_like(points)
    points = points.reshape(4,2)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew

def warpImg(img, points, width, height, pad=7):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width,height))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp