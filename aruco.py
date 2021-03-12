# TODO: Can have a canvas class which is composed of a list of arucos and can calculate distance between any two pixels
import numpy as np


class Aruco:

    def __init__(self, bbox: list, marker_id: int, width=10):
        """
        Creates a new aruco marker
        :param bbox: Bounding box of the aruco marker
        :param marker_id: id of aruco marker
        :param width: Width of aruco marker in mm
        """
        self.bbox = bbox
        self.id = marker_id
        self.width = width

    def get_top_left_point(self) -> list:
        return self.bbox[0]

    @staticmethod
    def point_to_point_dist(point1, point2) -> float:
        """
        Calculates the distance between two points in pixels
        :param point1: first point represented by an array of 2 elements
        :param point2: second point represented by an array of 2 elements
        :return: Distance between two points in pixels
        """
        np_point1 = np.asarray(point1)
        np_point2 = np.asarray(point2)
        return np.linalg.norm(np_point2 - np_point1)

    def mm_to_pix_ratio(self) -> float:
        """
        Calculates the mm / pix ratio based on the width of bbox
        :return: Returns the mm / pix ratio
        """
        return self.width / Aruco.point_to_point_dist(self.bbox[0], self.bbox[1])

    def dist_to_point(self, point: list) -> float:
        """
        Gets the distance from top left corner of self to point in pixels
        :param point: Point to calculate distance to
        :return: Returns the distance in pixels from self top left corner to point
        """
        # calculate the distance in terms of pixel, divide it by the width of the marker (px), multiply it by 10mm
        point1 = self.get_top_left_point()
        pixel_dist = Aruco.point_to_point_dist(point, point1)
        return pixel_dist * self.mm_to_pix_ratio()

    def __str__(self):
        return f"Aruco Object: id={self.id}, width={self.width}"
