# inspired by https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515
from tkinter import OUTSIDE
from turtle import position
import cv2
import math
import numpy as np
from enum import Enum
from Pointfunctions import doIntersect, line, intersection, angle
from classes.classes import Tracker, Vehicle, State


CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

CV_PI = 3.14

class VehicleCounter(Tracker):
    DIVIDER_COLOUR = (255, 255, 0)
    BOUNDING_BOX_COLOUR = (255, 0, 0)
    CENTROID_COLOUR = (0, 0, 255)
    def __init__(self, shape, divider, secondline, distance, fps=30):
        print("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider
        self.fps = fps
        self.secondline = secondline
        self.distance = distance # in m
        self.n_frame = 0
        self.frame = -1
        print(fps)

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 7

        self.vPointAvg = []



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

    @staticmethod
    def is_valid_vector(a):
        distance, angle = a
        threshold_distance = max(20., -0.008 * angle**2 + 0.4 * angle + 25.0)
        return (distance <= threshold_distance)


    def update_vehicle(self, vehicle, matches):
        # Find if any of the matches fits this vehicle
        for i, match in enumerate(matches):
            contour, centroid = match

            vector = self.get_vector(vehicle.last_position, centroid)
            if self.is_valid_vector(vector):
                vehicle.add_position(centroid)
                vehicle.set_vector(vector)
                print(f"Added match ({centroid[0]}, {centroid[1]}) to vehicle #{vehicle.id}. vector=({vector[0]:.2f},{vector[1]:.2f})")
                return i

        # No matches fit...        
        vehicle.frames_since_seen += 1
        print(f"No match for vehicle #{vehicle.id}. frames_since_seen={vehicle.frames_since_seen}")

        return None


    def update_count(self, matches, output_image, frame_number, write=False):
        cv2.line(output_image, self.divider[0], self.divider[1], self.DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
        cv2.line(output_image, self.secondline[0], self.secondline[1], self.DIVIDER_COLOUR, 1)
        print(f"Updating count using {len(matches)} matches...")

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
                print(matches[i])
                # cv2.rectangle(output_image, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
                # cv2.circle(output_image, centroid, 2, CENTROID_COLOUR, -1)
                del matches[i]

        # Add new vehicles based on the remaining matches
        for match in matches:
            contour, centroid = match
            new_vehicle = Vehicle(self.next_vehicle_id, centroid)
            self.next_vehicle_id += 1
            self.vehicles.append(new_vehicle)
            print(f"Created new vehicle #{new_vehicle.id} from match ({centroid[0]}, {centroid[1]}).")

        # Count any uncounted vehicles that are past the divider
        for vehicle in self.vehicles:
            if not vehicle.counted :#and abs(vehicle.last_position[1] - self.divider) < 20:
                if len(vehicle.positions) > 1:
                    twoLast = vehicle.positions[-2:]

                    state = 1 if doIntersect(twoLast, self.divider) else 2 if doIntersect(twoLast, self.secondline) else 0
                    if state :
                        if vehicle.state == State.OUTSIDE : # crossed a line
                            print(f"Vehicle {vehicle.id} passed the first line{State}")
                            vehicle.frame = frame_number
                            vehicle.state = state
                        elif vehicle.state != state: # crossed a different line
                            vehicle.counted = True
                            time = (frame_number - vehicle.frame) / self.fps # seconds
                            vehicle.speed = self.distance / time * 3.6 # m/s to km/h
                            
                            self.vehicle_count += 1
                            print(f"Vehicle {vehicle.id} passed the second line, avg speed {vehicle.speed} km/h")
                            print(f"Counted vehicle #{vehicle.id} (total count={self.vehicle_count}).")
                        else: # crossed the same line : went back, most likelly not normal ^^
                            vehicle.state = State.OUTSIDE
            if len(vehicle.direction) != 0 and len(self.vPointAvg) != 0 and not vehicle.counted:
                imgPointInf = self.vPointAvg[:2]
                an = angle([vehicle.last_position, imgPointInf], vehicle.direction)
                if abs(an) < 6: # angle between line from position to point at infinity and direction vector
                    l = line(vehicle.last_position, imgPointInf)
                    #l2 = line(vehicle.last_position, vehicle.direction)
                    p1 = intersection(line(self.divider[0], self.divider[1]), l)
                    p2 = intersection(line(self.secondline[0], self.secondline[1]), l)

                    # if the points are on the segments
                    if (self.divider[0][0] <= p1[0] <= self.divider[1][0] and self.divider[0][1] <= p1[1] <= self.divider[1][1] and # point 1 is on the divider
                        self.secondline[0][0] <= p2[0] <= self.secondline[1][0] and self.secondline[0][1] <= p2[1] <= self.secondline[1][1]): # point2 is on the second line

                        # colinear point a(car), b(marker1), c(marker2), d(vanishing point)
                        # from CR(a,b,c,d), d point at infinity (https://en.wikipedia.org/wiki/Cross-ratio#Definition)
                        # acp stands for A'B'
                        # so ac = acp*bdp*bc/(bcp*adp)
                        acp = math.dist(vehicle.last_position, p2)
                        bdp = math.dist(p1, imgPointInf)
                        bcp = math.dist(p1, p2)
                        adp = math.dist(vehicle.last_position, imgPointInf)
                        bc = self.distance
                        if bcp != 0 and adp !=0: 
                            ac = acp*bdp*bc/(bcp*adp)
                            
                            p1 = (round(p1[0]), round(p1[1]))
                            p2 = (round(p2[0]), round(p2[1]))

                            if len(vehicle.distanceMarkers) > 5: #take 3 before to have a better estimate
                                time = (frame_number - vehicle.distanceMarkers[-5][1]) / self.fps # seconds
                                speed = abs(ac - vehicle.distanceMarkers[-5][0]) / time * 3.6 # m/s to km/h
                                print(f"Vehicle {vehicle.id} CR Speed: {speed:.1f}")
                                cv2.putText(output_image, f"CR speed : {speed:.1f}", (vehicle.last_position[0] + 20, vehicle.last_position[1]), cv2.FONT_HERSHEY_PLAIN, 1, vehicle.car_colour)
                            vehicle.distanceMarkers.append([ac, frame_number])
                            cv2.circle(output_image, p1, 2, (150,150,150), -1)
                            cv2.circle(output_image, p2, 2, (150,150,150), -1)
                            print("------ ac", ac, "angle", an,"Points 1:", p1, "2:", p2,self.vPointAvg)
                else :
                    print("angle refused:", an)


                # vehicle.lineTrack(output_image)

        for i in range(len(self.vehicles)):
            if self.vehicles[i].counted:
                self.vehicles[i].lineTrack(output_image)
                if i > 0:
                    try:
                        v = np.cross(self.vehicles[i-1].line[-1], self.vehicles[i].line[-1])
                        v = v/v[2]
                    except:
                        continue
                    self.vPointAvg = v
        #print('Vanishing point with cars: ', self.vPointAvg)

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)
            
            cv2.putText(output_image, (f"{self.vehicle_count:.2f}"), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 1)

        # Remove vehicles that have not been seen long enough
        # removed = [ v.id for v in self.vehicles
        #     if v.frames_since_seen >= self.max_unseen_frames ]
        removed = [ v for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]
        for v in removed:
            # v.lineTrack(output_image)
            print(f"Removed vehicle #{v.id}.")

        print(f"Count updated, tracking {len(self.vehicles)} vehicles.")
