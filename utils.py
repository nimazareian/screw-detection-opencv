import cv2
import numpy as np

def empty(val):
    pass

def getContours(img, cannyThr=[200,200], gaussianBlur=(5,5), showCanny=False, minArea=1000, maxArea=1000000, filterEdges = 0, drawContours=False):
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
        if minArea < area < maxArea:
            peri = cv2.arcLength(i,True) # True for contour being closed
            approx = cv2.approxPolyDP(i, 0.02 * peri, True) # Approximation of edges of contour
            bbox = cv2.boundingRect(approx) # Bounding box
            print("bbox:", bbox)
            if filterEdges > 0:
                if len(approx) == filterEdges:
                    finalContours.append([len(approx), area, approx, bbox, i])
            else:
                # if filterEdges set to 0, don't filter based on edge count
                finalContours.append([len(approx), area, approx, bbox, i])

    finalContours = sorted(finalContours, key = lambda x: x[1], reverse=True) # Sorting based on area (largest first)
    if drawContours:
        for contour in finalContours:
            cv2.drawContours(img, contour[4], -1, (0,0,255), 3)

    return img, finalContours

def reorder(points):
    pointsNew = np.zeros_like(points)
    points = points.reshape(4,2)
    add = points.sum(1)
    pointsNew[0] = points[np.argmin(add)]
    pointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    pointsNew[1] = points[np.argmin(diff)]
    pointsNew[2] = points[np.argmax(diff)]
    return pointsNew


def drawBoxAroundContour(img, contour, scale=1, drawMinAreaRect=True, drawBoundingRect=False, drawPolylines=False, drawDimensions=True, drawArea=False, offsetDimension=0):
    rect = cv2.minAreaRect(contour[2])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if drawMinAreaRect: cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    nPoints = reorder(box)
    screw_w = round(findDist(nPoints[0], nPoints[1]) / scale, 3)
    screw_h = round(findDist(nPoints[0], nPoints[2]) / scale, 3)
    maxLength = max(screw_w, screw_h) + offsetDimension
    if drawBoundingRect:
        x, y, w, h = cv2.boundingRect(contour[2]) # Straight rectangle Around contour
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255, 0), 2)

    if drawPolylines: cv2.polylines(img, [contour[2]], True, (0,255,0), 2) # Polylines around contour - True for closed

    x, y, w, h = contour[3]
    if drawArea:
        cv2.putText(img, '{} area'.format(w * h), (x, y + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    if drawDimensions:
        # cv2.putText(img, '{} in'.format(screw_h), (x, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
        roundedLength = round(maxLength, 3)
        cv2.putText(img, '{} in'.format(roundedLength), (x, y + 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 1)
    # TODO: Return the position of the box aswell
    return maxLength


def warpImg(img, points, width, height, pad=7):
    # print(points)
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (width,height))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    return imgWarp

def findDist(pts1, pts2):
    return ((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1])**2 )**0.5