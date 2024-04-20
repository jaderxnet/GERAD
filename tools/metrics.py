from enum import Enum
import numpy as np


class MetricTipe(Enum):
    GENERAL = 1
    EPDNVP = 2
    EPDNM = 3
    EPDNMVP = 4
    EPE = 5


class Metric:
    def __init__(self, metricName, metricTipe=MetricTipe.GENERAL, screenDimentions=(1920, 1080)) -> None:
        self.metricName = metricName
        self.metricTipe = metricTipe
        self.metricValue = 0
        self.screenDimentions = screenDimentions
        self.minor_distance = 100000
        self.index = -1

    def increaseValue(self):
        self.metricValue += self.minor_distance

    def normalize_distance_visible(self, keypoints1, keypoints2):
        sum_distance = 0
        for point1, point2 in zip(keypoints1, keypoints2):
            dist = np.linalg.norm(point1[0:2]-point2[0:2])
            if (len(point1) > 2 and len(point2) > 2):
                dist *= point1[-1]*point2[-1]
            sum_distance += dist
        return sum_distance

    def normalize_distance_media(self, keypoints1, keypoints2):
        sum_distance = 0
        for point1, point2 in zip(keypoints1, keypoints2):
            dist = np.linalg.norm(point1[0:2]-point2[0:2])
            sum_distance += dist
        sum_distance /= 17
        return sum_distance

    def average_pixel_distance(self, keypoints1, keypoints2):
        averege = 0
        count = 0
        # print("Keys 1:", keypoints1, " Keys 2: ", keypoints2)
        for point1, point2 in zip(keypoints1, keypoints2):
            dist = np.linalg.norm(point1[0:2] - point2[0:2])
            # print("dist ", dist)
            averege += dist
            count += 1
            # print("average ", count, " - ", averege)
        averege = averege/17
        # print("average /17 - ", averege)
        return averege

    def updateMinorDistance(self, keypoints1="NAN", keypoints2="NAN", index="NAN", otherDistance="NAN"):
        if (self.metricTipe == MetricTipe.EPDNVP):
            distance = self.normalize_distance_visible(keypoints1, keypoints2)
            if (self.minor_distance > distance):
                self.minor_distance = distance
                self.index = index
        elif (self.metricTipe == MetricTipe.EPDNMVP):
            distance = otherDistance/17
            if (self.minor_distance > distance):
                self.minor_distance = distance
                self.index = index
        elif (self.metricTipe == MetricTipe.EPDNM):
            distance = self.normalize_distance_media(keypoints1, keypoints2)
            if (self.minor_distance > distance):
                self.minor_distance = distance
                self.index = index
        elif (self.metricTipe == MetricTipe.EPE):
            distance = self.average_pixel_distance(self.to_pixel_coords(keypoints1),
                                                   self.to_pixel_coords(keypoints2))
            if (self.minor_distance > distance):
                self.minor_distance = distance
                self.index = index
        return distance

    def to_pixel_coords(self, relative_coords):
        # print("to_pixel_coords: ", relative_coords)
        result = []
        for coord in relative_coords:
            result.append(coord[1:3] * self.screenDimentions)
        return result
