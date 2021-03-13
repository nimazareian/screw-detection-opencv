import cv2
import cv2.aruco as aruco
import numpy as np
import os
from custom_aruco import CustomAruco


camera_matrix = np.loadtxt('camera_matrix.txt', delimiter=',')
camera_distortion = np.loadtxt('camera_distortion.txt', delimiter=',')


def find_aruco_markers(img, marker_size=6, total_markers=250, draw=True):
    """Finds all arucos in img

    :param img: Image to look through
    :param marker_size: Number of bits per side of marker, defaults to 6
    :param total_markers: total number of markers that compose the dictionary, defaults to 250
    :param draw: If updated img should be drawn, defaults to True
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()
    bboxs, ids, rejected_bboxs = aruco.detectMarkers(img_gray, aruco_dict,
                                                     parameters=aruco_param, cameraMatrix=camera_matrix, distCoeff=camera_distortion)

    # print(bboxs)
    if draw:
        aruco.drawDetectedMarkers(img, bboxs, ids)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners=bboxs, markerLength=2, cameraMatrix=camera_matrix, distCoeffs=camera_distortion)
        for rvec, tvec in zip(rvecs, tvecs):
            aurco.drawAxis(img, cameraMatrix=camera_matrix,
                           distCoeffs=camera_distortion, rvec=rvec, tvec=tvec, length=1)
    return bboxs, ids

# TODO: Chess board square size 0.988"


def main():
    cap = cv2.VideoCapture(0)

    while True:
        # 48 mm width
        img = cv2.imread('resources/aruco_example_multiple_1.png')
        # success, img = cap.read()
        bboxs, ids = find_aruco_markers(img)
        # a = CustomAruco(bboxs[0], ids[0], 10)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
