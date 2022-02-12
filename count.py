import cv2
from cv2 import imshow
import numpy as np

import vehicleCounter

c = cv2.VideoCapture("video.mp4")


_,f2 = c.read()
f2gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

count = 0

vehicles = []

DIVIDER_COLOUR = (255, 255, 0)
BOUNDING_BOX_COLOUR = (255, 0, 0)
CENTROID_COLOUR = (0, 0, 255)

frame_number = 1
car_counter = vehicleCounter.VehicleCounter(f2.shape[:2], f2.shape[0] / 4, 3*f2.shape[0] / 4)#, c.get(cv2.CAP_PROP_FPS))


while(c.isOpened()):
    frame_number += 1
    _,f = c.read()
    fgray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(fgray, f2gray)

    processed = f.copy()
    print(fgray.shape, processed.shape, car_counter.divider)
    cv2.line(processed, (0, int(car_counter.divider)), (fgray.shape[1], int(car_counter.divider)), DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
    cv2.line(processed, (0, int(car_counter.secondline)), (fgray.shape[1], int(car_counter.secondline)), DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
    th, dframe = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    dilate_frame = cv2.dilate(dframe, None, iterations=10)

    contours, hierarchy = cv2.findContours(dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    matches = []
    for contour in contours:
        # continue through the loop if contour area is less than 500...
        # ... helps in removing noise detection
        if cv2.contourArea(contour) < 500:
            continue
        # get the xmin, ymin, width, and height coordinates from the contours
        (x, y, w, h) = cv2.boundingRect(contour)

        centroid = ((x+w//2, y+h//2))
        matches.append([(x, y, w, h), centroid])

        cv2.rectangle(processed, (x, y), (x + w - 1, y + h - 1), BOUNDING_BOX_COLOUR, 1)
        cv2.circle(processed, centroid, 2, CENTROID_COLOUR, -1)

    car_counter.update_count(matches, processed)

    cv2.imshow('Detected Objects', processed)
    
    k = cv2.waitKey()#20

    if k == 27:
        break
    f2gray = fgray

c.release()
cv2.destroyAllWindows()