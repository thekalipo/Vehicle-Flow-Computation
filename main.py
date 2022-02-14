import json
import cv2
from cv2 import imshow
import numpy as np
from classes.frameSubDetector import FrameSubDetector
from classes.noProcessor import noProcessor
from classes.sortTracker import sortTracker
from classes.yoloFaster import FasterYoloDetector

from classes.vehicleCounter import VehicleCounter

from classes.classes import *

import sys
video = sys.argv[1] if len(sys.argv) > 1 else 0

if video == '1':
    video_name = "motorway.mov"
    line1 = [[617, 1100], [2970, 1075]] # [[617, 1300], [2970, 1275]]
    line2 = [[76, 1304], [3601, 1267]] #[[76, 1704], [3601, 1667]]
    detector = FrameSubDetector() # Detector
elif video == '2':
    # 1262, 1478
    video_name = "live.mov"
    line1 = [[216,500], [1153, 400]]
    line2 = [[218, 700], [1262, 550]]
    detector = FasterYoloDetector() # Detector
else:
    video_name = "video.mp4"
    line1 = [[316,63], [453, 75]]
    line2 = [[218, 201], [529, 231]]
    detector = FrameSubDetector() # Detector
    
# c = cv2.VideoCapture("motorway.mov")
c = cv2.VideoCapture(video_name)

FPS = c.get(cv2.CAP_PROP_FPS)
_,f = c.read()

frame_number = 0

processor = noProcessor() # processor

distance = 27.43 #distance in m between the two lines
# 10 feet, and the empty spaces in-between measure 30 feet, in our case must be in metters, so 40+40+10 => 27.43m
tracker = sortTracker(f.shape[:2], line2, line1, 24.3, FPS)
#tracker = VehicleCounter(f.shape[:2], line2, line1, 24.3, FPS) #Tracker

OUTPUT_EVERY = 5 # every 15 secs
every = round(OUTPUT_EVERY * FPS)

while(c.isOpened()):
    frame_number += 1

    _,f = c.read()
    if f is None:
        break
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