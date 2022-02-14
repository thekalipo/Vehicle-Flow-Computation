from classes.classes import Tracker
from classes.sort import Sort
import numpy as np
import cv2
class sortTracker(Tracker):
    def __init__(self):
        self.mot_tracker = Sort()

    def update_count(self, matches, processed, frame_number):
        print (matches)
        boxes = np.array([b for b,_ in matches])
        boxes = np.array([list(b[:2])+[b[2]+b[0], b[3]+b[1]] for b,_ in matches])
        mot_ids = self.mot_tracker.update(boxes if len(boxes) else np.empty((0,5)))
        print(boxes)
        for b in mot_ids:
            cv2.rectangle(processed, tuple(b[:2]), tuple(b[2:4]), (255, 0, 0))
            cv2.putText(processed, str(b[4]), (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        