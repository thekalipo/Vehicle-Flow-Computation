import json
import cv2
from cv2 import imshow
import numpy as np
from classes.frameSubDetector import FrameSubDetector
from classes.noProcessor import noProcessor
from classes.sortTracker import sortTracker
from classes.yoloFaster import FasterYoloDetector

import vehicleCounter

from classes.classes import *

c = cv2.VideoCapture("video.mp4")
FPS = c.get(cv2.CAP_PROP_FPS)
_,f = c.read()

frame_number = 0

line1 = [[316,63], [453, 75]]
line2 = [[218, 201], [529, 231]]


processor = noProcessor() # processor
detector = FrameSubDetector() # Detector
#detector = FasterYoloDetector() # Detector

distance = 27.43 #distance in m between the two lines
# 10 feet, and the empty spaces in-between measure 30 feet, in our case must be in metters, so 40+40+10 => 27.43m
tracker = sortTracker(f.shape[:2], line2, line1, 24.3, FPS)
#tracker = vehicleCounter.VehicleCounter(f.shape[:2], line2, line1, 24.3, FPS) #Tracker

OUTPUT_EVERY = 15 # every 15 secs
every = round(OUTPUT_EVERY * FPS)

while(c.isOpened()):
    frame_number += 1

    _,f = c.read()

    processed = processor.process(f.copy())

    matches = detector.findMatches(processed)

    if frame_number % every == 0:
        v = tracker.update_count(matches, processed, frame_number, True)
    else:
        v = tracker.update_count(matches, processed, frame_number)

    cv2.imshow('Image', processed)
    
    k = cv2.waitKey(10)#20

    if k == 27:
        break

c.release()
cv2.destroyAllWindows()