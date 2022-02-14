from abc import ABC, abstractmethod
import math
import cv2
import numpy as np
from enum import Enum

class Detector(ABC):
    @abstractmethod
    def findMatches(self, frame):
        pass

class Processor(ABC):
    @abstractmethod
    def process(self, frame):
        pass

class Tracker(ABC):
    @abstractmethod
    def update_count(self, matches, processed, frame_number):
        pass

    @staticmethod
    def get_vector(a, b):
        """Calculate vector (distance, angle in degrees) from point a to point b.

        Angle ranges from -180 to 180 degrees.
        Vector with angle 0 points straight down on the image.
        Values increase in clockwise direction.
        """
        dx = float(b[0] - a[0])
        dy = float(b[1] - a[1])

        distance = math.sqrt(dx**2 + dy**2)

        if dy > 0:
            angle = math.degrees(math.atan(-dx/dy))
        elif dy == 0:
            if dx < 0:
                angle = 90.0
            elif dx > 0:
                angle = -90.0
            else:
                angle = 0.0
        else:
            if dx < 0:
                angle = 180 - math.degrees(math.atan(dx/dy))
            elif dx > 0:
                angle = -180 - math.degrees(math.atan(dx/dy))
            else:
                angle = 180.0        

        return distance, angle 

class State(Enum): # to check for cars in the two ways
    OUTSIDE = 0
    FIRSTLINE = 1
    SECONDLINE = 2

CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

CV_PI = 3.14

class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted = False
        self.angles = []
        self.direction = []
        self.vector = (0,0) # distance, angle
        self.avg_vector = (0,0,0) # distance, angle, number
        self.state = State.OUTSIDE
        self.speed = 0
        self.distanceMarkers = [] 
        self.line = []

        self.distance = 0
        self.real_positions = []

        self.car_colour = (0,0,0)

    @property
    def last_position(self):
        return self.positions[-1]
    
    @property
    def first_position(self):
        return self.positions[0]

    @staticmethod
    def circ_mean(angles):
        """angles in degrees"
        """
        cosList = []
        cosSum = 0.0
        sinSum = 0.0
        for i in angles:
            cosSum += math.cos(math.radians(float(i)))
            sinSum += math.sin(math.radians(float(i)))
        N = len(angles)
        C = cosSum/N
        S = sinSum/N
        theMean = math.atan2(S,C)
        if theMean < 0.0:
            theMean += math.radians(360.0)
        return math.degrees(theMean)

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
    
    def set_vector(self, vector):
        self.vector = (vector[0], vector[1])
        self.angles.append(self.vector[1]+90)
        self.avg_vector = ((self.avg_vector[0] * self.avg_vector[2] + self.vector[0]) / (self.avg_vector[2] + 1), 
                          (self.avg_vector[1] * self.avg_vector[2] + self.vector[1]) / (self.avg_vector[2] + 1), 
                          self.avg_vector[2] + 1)
        print(f"angle {self.avg_vector[2]}", self.vector, self.avg_vector)

    def draw(self, output_image):
        self.car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, self.car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, self.car_colour, 1)
        if len(self.positions) > 2:
            last = self.positions[-1]
            #dist = self.vector[0] * 4
            dist = self.avg_vector[0] * 4
            angle = self.circ_mean(self.angles) #self.avg_vector[1]
            x =  round(last[0] + dist * math.cos(angle * CV_PI / 180.0))
            y =  round(last[1] + dist * math.sin(angle * CV_PI / 180.0))
            self.direction = [last, (x,y)]
            #print(x,y)
            cv2.arrowedLine(output_image,last, (x, y), self.car_colour, 2)
            #cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10), cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 1)
            if self.counted :
                cv2.putText(output_image, f"{self.speed:.1f}", (last[0] - 10, last[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, self.car_colour)

    def lineTrack(self, output_image):
        if len(self.positions) > 8:
            # if self.counted:
            a = [self.first_position[0], self.first_position[1], 1]
            b = [self.last_position[0], self.last_position[1], 1]
            self.line.append(np.cross(a, b))

            cv2.line(output_image, (a[0], a[1]), (b[0], b[1]), (0,0,255), 1)