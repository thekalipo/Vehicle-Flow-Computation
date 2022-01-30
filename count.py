import cv2
from cv2 import imshow
import numpy as np

c = cv2.VideoCapture("video.mp4")


_,f2 = c.read()
f2gray = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

count = 0

vehicles = []


def distance(x1,x2,y1,y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def search_vechicle(x,y):
    for i,v in enumerate(vehicles):
        dist = distance(x, v[0], y, v[1])
        if dist < 40:
            vehicles[i] = (x,y)
            print(f"found: ({x},{y}) index", i, "",vehicles)
            return True
    vehicles.append((x,y))
    print(False)
    return False

def delVehicle(x,y):
    for v in vehicles:
        dist = distance(x, v[0], y, v[1])
        if dist < 40:
            vehicles.remove(v)
            print("removed :", vehicles)

while(c.isOpened()):
    _,f = c.read()
    fgray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    diff = cv2.absdiff(fgray, f2gray)
    th, dframe = cv2.threshold(diff, 100, 255, cv2.THRESH_BINARY)
    dilate_frame = cv2.dilate(dframe, None, iterations=5)

    contours, hierarchy = cv2.findContours(dilate_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, cnt in enumerate(contours):
        cv2.drawContours(f, contours, i, (0, 0, 255), 3)

    
    for contour in contours:
        # continue through the loop if contour area is less than 500...
        # ... helps in removing noise detection
        if cv2.contourArea(contour) < 1000:
            continue
        # get the xmin, ymin, width, and height coordinates from the contours
        (x, y, w, h) = cv2.boundingRect(contour)

        if 155<=y<=165 and search_vechicle(x+w/2, y+h/2):
            count += 1
        elif y > 165:
            delVehicle(x+w/2, y+h/2)
        # draw the bounding boxes
        cv2.rectangle(f, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.line(f, (0,160), (640, 160), (255,0,0), 2)
    cv2.putText(f,"count : "+str(count), (10,50), 0, 1, 255,2)

    cv2.imshow('Detected Objects', f)

    #imshow("diff", dilate_frame)
    k = cv2.waitKey(20)

    if k == 27:
        break
    f2gray = fgray

c.release()
cv2.destroyAllWindows()