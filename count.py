import cv2
from cv2 import imshow
import numpy as np

import vehicleCounter

c = cv2.VideoCapture("video.mp4")
# c = cv2.VideoCapture("motorway.mov")


_,f2 = c.read()

# h2 = f2.shape[0]
# w2 = f2.shape[1]

# src2 = np.array([[350, 0], [435, 0], [w2, h2], [30, h2]], np.float32)
# dst2 = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], np.float32)


# M2 = cv2.getPerspectiveTransform(src2, dst2)
# warped = cv2.warpPerspective(f2, M2, (int(w2), int(h2)))

# f2gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
f2gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

count = 0

vehicles = []



frame_number = 0
#line1 = [(0,int(f2.shape[0] / 4)), (f2.shape[1], int(f2.shape[0] / 4))]
#line2 = [(0, int(3*f2.shape[0] / 4)), (f2.shape[1], int(3*f2.shape[0] / 4))]

# line1 = [[617, 1300], [2970, 1275]]
# line2 = [[76, 1704], [3601, 1667]]

# line1 = [[80, 220], [555, 235]]
# line2 = [[75, 330], [590, 340]]

p1 = [361,68]
p2 = [320, 212]

line1 = [[316,63], [453, 75]]
line2 = [[218, 201], [529, 231]]

#line1 = [[670,375],[718,475]]
#line2 = [[968,500],[917,370]]

#line1 = [[617, 1300], [2970, 1275]]
#line2 = [[76, 1704], [3601, 1667]]

distance = 27.43 # 10 feet, and the empty spaces in-between measure 30 feet, in our case must be in metters, so 40+40+10 => 27.43m
car_counter = vehicleCounter.VehicleCounter(f2.shape[:2], line2, line1, 24.3, c.get(cv2.CAP_PROP_FPS), point1 = p1, point2 = p2) 

#cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL) 

while(c.isOpened()):
    frame_number += 1
    _,f = c.read()

    # h = f.shape[0]
    # w = f.shape[1]
    # 360, 640

    # src = np.array([[350, 0], [435, 0], [w, h], [30, h]], np.float32)
    # dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    # M = cv2.getPerspectiveTransform(src, dst)
    # warped = cv2.warpPerspective(f, M, (int(w), int(h)))
    
    # fgray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    # processed = warped.copy()
    
    fgray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(fgray, f2gray)

    processed = f.copy()
    print(fgray.shape, processed.shape, car_counter.divider)
    cv2.line(processed, car_counter.divider[0], car_counter.divider[1], DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
    cv2.line(processed, car_counter.secondline[0], car_counter.secondline[1], DIVIDER_COLOUR, 1) # calcOpticalFlowFarneback ?
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

    car_counter.update_count(matches, processed, frame_number)

    cv2.imshow('Detected Objects', processed)
    
    k = cv2.waitKey(20)#20

    if k == 27:
        break
    f2gray = fgray

c.release()
cv2.destroyAllWindows()