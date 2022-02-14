from classes.classes import Tracker, State, Vehicle
from classes.sort import Sort
import numpy as np
import cv2
from Pointfunctions import doIntersect, line, intersection, angle
import math
import json

CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

CV_PI = 3.14

class sortTracker(Tracker):
    DIVIDER_COLOUR = (255, 255, 0)
    BOUNDING_BOX_COLOUR = (255, 0, 0)
    CENTROID_COLOUR = (0, 0, 255)
    def __init__(self, shape, divider, secondline, distance, fps=30):
        self.mot_tracker = Sort()
        self.vehicles = {}

        self.height, self.width = shape
        self.divider = divider
        self.fps = fps
        self.secondline = secondline
        self.distance = distance # in m
        self.frame = -1
        print(fps)

        self.vehicle_count = 0

        self.vPointAvgs = []
        self.vPointAvg = []
    

    def update_count(self, matches, output_image, frame_number, write=False):
        #print (matches)
        boxes = np.array([b for b,_ in matches])
        boxes = np.array([list(b[:2])+[b[2]+b[0], b[3]+b[1]] for b,_ in matches])
        mot_ids = self.mot_tracker.update(boxes if len(boxes) else np.empty((0,5)))
        #print(boxes)
        for b in mot_ids:
            #print(b)
            id = b[-1]
            centroid = [(b[0]+(b[2]-b[0])//2), (b[1]+(b[3]-b[1])//2)]
            if id not in self.vehicles:
                print("new vehicle", id)
                self.vehicles[id] = Vehicle(id, centroid)
            else :
                vehicle = self.vehicles[id]
                vector = self.get_vector(vehicle.last_position, centroid)
                #print(vector)
                vehicle.add_position(centroid)
                vehicle.set_vector(vector)
                vehicle.draw(output_image)
            
            #cv2.rectangle(output_image, tuple(b[:2]), tuple(b[2:4]), (255, 0, 0))
            #cv2.putText(output_image, str(b[4]), (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        vehicles = list(self.vehicles.values())
        # Count any uncounted vehicles that are past the divider
        for vehicle in vehicles:
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
                                print("Vehicle {vehicle.id} CR Speed: ",speed)
                                cv2.putText(output_image, f"CR speed : {speed:.1f}", (vehicle.last_position[0] + 20, vehicle.last_position[1]), cv2.FONT_HERSHEY_PLAIN, 1, vehicle.car_colour)
                            vehicle.distanceMarkers.append([ac, frame_number])
                            cv2.circle(output_image, p1, 2, (150,150,150), -1)
                            cv2.circle(output_image, p2, 2, (150,150,150), -1)
        
        for i in range(len(self.vehicles)):
            try:
                self.vehicles[i].lineTrack(output_image)
            except:
                continue
            if i > 0:
                try:
                    v = np.cross(self.vehicles[i-1].line[-1], self.vehicles[i].line[-1])
                    v = v/v[2]
                except:
                    continue
                if not np.isnan(v[0]):
                    self.vPointAvgs.append(v)
                # self.vPointAvg = self.vPointAvgs[-1]
                # print('VANISHING POINT: ', self.vPointAvg)
                # cv2.circle(output_image, (int(self.vPointAvg[0]), int(self.vPointAvg[1])), 10, (255,255,0), -1)

        if len(self.vPointAvgs) > 2:
            self.vPointAvg = self.vPointAvgs[-2]
            #print('VANISHING POINT: ', self.vPointAvg)
        elif len(self.vPointAvgs) == 1:
            self.vPointAvg = self.vPointAvgs[-1]
            #print('VANISHING POINT: ', self.vPointAvg)
        
        cv2.putText(output_image, (f"{self.vehicle_count:.2f}"), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 1)
        cv2.line(output_image, self.divider[0], self.divider[1], self.DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
        cv2.line(output_image, self.secondline[0], self.secondline[1], self.DIVIDER_COLOUR, 1)
        if write:
            with open(f"log{frame_number//self.fps}.txt", "w") as l:
                json.dump({frame_number:[{"id":str(i), "avgSpeed":str(self.vehicles[i].speed)} for i in self.vehicles if self.vehicles[i].counted]}, l)
            # delete old ones
            ids = [int(i[-1]) for i in mot_ids]
            self.vehicles = { key:value for (key,value) in self.vehicles.items() if key in ids}