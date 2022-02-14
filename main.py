import cv2
from cv2 import imshow
import numpy as np
from classes.frameSubDetector import FrameSubDetector
from classes.noProcessor import noProcessor

import vehicleCounter

from classes.classes import *



c = cv2.VideoCapture("video.mp4")

_,f = c.read()

frame_number = 0

p1 = [361,68]
p2 = [320, 212]

line1 = [[316,63], [453, 75]]
line2 = [[218, 201], [529, 231]]


distance = 27.43 # 10 feet, and the empty spaces in-between measure 30 feet, in our case must be in metters, so 40+40+10 => 27.43m
tracker = vehicleCounter.VehicleCounter(f.shape[:2], line2, line1, 24.3, c.get(cv2.CAP_PROP_FPS), p1, p2) #Tracker
processor = noProcessor() # processor
detector = FrameSubDetector() # Detector


while(c.isOpened()):
    frame_number += 1
    _,f = c.read()
    processed = processor.process(f.copy())

    matches = detector.findMatches(processed)

    tracker.update_count(matches, processed, frame_number)

    cv2.imshow('Image', processed)
    
    k = cv2.waitKey(20)#20

    if k == 27:
        break

c.release()
cv2.destroyAllWindows()