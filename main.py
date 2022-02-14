import cv2
from cv2 import imshow
import numpy as np
from classes.frameSubDetector import FrameSubDetector
from classes.noProcessor import noProcessor
from classes.sortTracker import sortTracker
from classes.yoloFaster import FasterYoloDetector

import vehicleCounter

from classes.classes import *

import sys
video = sys.argv[1]

if video == '0':
    video_name = "video.mp4"
    line1 = [[316,63], [453, 75]]
    line2 = [[218, 201], [529, 231]]
    detector = FrameSubDetector() # Detector
elif video == '1':
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
    
# c = cv2.VideoCapture("motorway.mov")
c = cv2.VideoCapture(video_name)

_,f = c.read()

frame_number = 0

processor = noProcessor() # processor

distance = 27.43 # 10 feet, and the empty spaces in-between measure 30 feet, in our case must be in metters, so 40+40+10 => 27.43m
# distance = 5
tracker = sortTracker(f.shape[:2], line2, line1, 24.3, c.get(cv2.CAP_PROP_FPS))
#tracker = vehicleCounter.VehicleCounter(f.shape[:2], line2, line1, 24.3, c.get(cv2.CAP_PROP_FPS)) #Tracker

while(c.isOpened()):
    frame_number += 1
    _,f = c.read()
    processed = processor.process(f.copy())

    matches = detector.findMatches(processed)

    tracker.update_count(matches, processed, frame_number)

    cv2.imshow('Image', processed)
    
    k = cv2.waitKey(10)#20

    if k == 27:
        break

c.release()
cv2.destroyAllWindows()