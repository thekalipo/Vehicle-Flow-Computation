# inspired by https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515
from turtle import position
import cv2
import math
import numpy as np



CAR_COLOURS = [ (0,0,255), (0,106,255), (0,216,255), (0,255,182), (0,255,76)
    , (144,255,0), (255,255,0), (255,148,0), (255,0,178), (220,0,255) ]

CV_PI = 3.14
class Vehicle(object):
    def __init__(self, id, position):
        self.id = id
        self.positions = [position]
        self.frames_since_seen = 0
        self.counted = False
        self.vector = (0,0)

    @property
    def last_position(self):
        return self.positions[-1]

    def add_position(self, new_position):
        self.positions.append(new_position)
        self.frames_since_seen = 0
    
    def set_vector(self, vector):
        self.vector = (vector[0], vector[1] + 90)
        print("angle", vector)

    def draw(self, output_image):
        car_colour = CAR_COLOURS[self.id % len(CAR_COLOURS)]
        for point in self.positions:
            cv2.circle(output_image, point, 2, car_colour, -1)
            cv2.polylines(output_image, [np.int32(self.positions)]
                , False, car_colour, 1)
        if len(self.positions) > 2:
            last = self.positions[-1]
            dist = self.vector[0] * 4
            x =  round(last[0] + dist * math.cos(self.vector[1] * CV_PI / 180.0))
            y =  round(last[1] + dist * math.sin(self.vector[1] * CV_PI / 180.0))
            #print(x,y)
            cv2.arrowedLine(output_image,last, (x,y), car_colour, 3)



# ============================================================================

class VehicleCounter(object):
    def __init__(self, shape, divider):
        print("vehicle_counter")

        self.height, self.width = shape
        self.divider = divider

        self.vehicles = []
        self.next_vehicle_id = 0
        self.vehicle_count = 0
        self.max_unseen_frames = 7


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


    def update_count(self, matches, output_image = None):
        print(f"Updating count using {len(matches)} matches...")

        # First update all the existing vehicles
        for vehicle in self.vehicles:
            i = self.update_vehicle(vehicle, matches)
            if i is not None:
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
            if not vehicle.counted and abs(vehicle.last_position[1] - self.divider) < 20:
                self.vehicle_count += 1
                vehicle.counted = True
                print(f"Counted vehicle #{vehicle.id} (total count={self.vehicle_count}).")

        # Optionally draw the vehicles on an image
        if output_image is not None:
            for vehicle in self.vehicles:
                vehicle.draw(output_image)

            cv2.putText(output_image, ("%02d" % self.vehicle_count), (142, 10)
                , cv2.FONT_HERSHEY_PLAIN, 1, (127, 255, 255), 1)

        # Remove vehicles that have not been seen long enough
        removed = [ v.id for v in self.vehicles
            if v.frames_since_seen >= self.max_unseen_frames ]
        self.vehicles[:] = [ v for v in self.vehicles
            if not v.frames_since_seen >= self.max_unseen_frames ]
        for id in removed:
            print(f"Removed vehicle #{id}.")

        print(f"Count updated, tracking {len(self.vehicles)} vehicles.")

# ============================================================================