"""
# YOLO object detection
import cv2 as cv
import numpy as np
import time

img = cv.imread('cars.jpg')
cv.imshow('window',  img)
cv.waitKey(1)

# Load names of classes and get random colors
classes = open('coco2.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Give the configuration and weight files for the model and load the network.
net = cv.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# determine the output layer
ln = net.getLayerNames()
print(net.getUnconnectedOutLayers()[:10])
print(ln[226])
print(ln[253])
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

# construct a blob from the image
blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
r = blob[0, 0, :, :]

cv.imshow('blob', r)
text = f'Blob shape={blob.shape}'
cv.displayOverlay('blob', text)
cv.waitKey(1)

net.setInput(blob)
t0 = time.time()
outputs = net.forward(ln)
t = time.time()
print('time=', t-t0)

print(len(outputs))
for out in outputs:
    print(out.shape)

def trackbar2(x):
    confidence = x/100
    r = r0.copy()
    for output in np.vstack(outputs):
        if output[4] > confidence:
            x, y, w, h = output[:4]
            p0 = int((x-w/2)*416), int((y-h/2)*416)
            p1 = int((x+w/2)*416), int((y+h/2)*416)
            cv.rectangle(r, p0, p1, 1, 1)
    cv.imshow('blob', r)
    text = f'Bbox confidence={confidence}'
    cv.displayOverlay('blob', text)

r0 = blob[0, 0, :, :]
r = r0.copy()
cv.imshow('blob', r)
cv.createTrackbar('confidence', 'blob', 50, 101, trackbar2)
trackbar2(50)

boxes = []
confidences = []
classIDs = []
h, w = img.shape[:2]

for output in outputs:
    for detection in output:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))
            classIDs.append(classID)

indices = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in colors[classIDs[i]]]
        cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
        cv.putText(img, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

cv.imshow('window', img)
cv.waitKey(0)
cv.destroyAllWindows()
"""

import cv2
import time
import sort
import numpy as np
import vehicleCounter

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.2
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

def accepted_classes(classid):
    return classid in [2,3,5,7]

mot_tracker = sort.Sort() 

vc = cv2.VideoCapture("video.mp4")

net = cv2.dnn.readNet("yolo-fastest-xl.weights", "yolo-fastest-xl.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

car_counter = vehicleCounter.VehicleCounter((640,360), 180)#f2.shape[0] / 2)

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    matches = []
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        if classid < len(class_names):
            label = "%s : %f" % (class_names[classid], score)
            cv2.rectangle(frame, box, color, 2)
            matches.append([box,(box[0]+box[2]//2, box[1]+box[3]//2)])
        else :
            print(classid, score)
    car_counter.update_count(matches, frame)
    end_drawing = time.time()

    #car_counter.update_count()
    
    #print(boxes,"||",scores)
    #print(boxes)
    """
    boxes2 = np.array([list(b[:2])+[b[2]+b[0], b[3]+b[1]] for i,b in enumerate(boxes)])
    #print("boxes",boxes2)
    #continue
    # print(scores)
    mot_ids = mot_tracker.update(boxes2 if len(scores) else np.empty((0, 5)))
    #print(mot_ids)
    start_drawing = time.time()
    matches = []
    for b in mot_ids:
        #print(tuple(b[:2]))
        #print(tuple(b[2:4]))
        #matches.append(tuple(b[:4]), )
        cv2.rectangle(frame, tuple(b[:2]), tuple(b[2:4]), COLORS[0])
        cv2.putText(frame, str(b[4]), (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
    end_drawing = time.time()
    print(mot_ids)
    """

    """
    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        if classid < len(class_names) and accepted_classes(classid):
            label = "%s : %f" % (class_names[classid], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else :
            print(classid, score)
    end_drawing = time.time()
    """
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.imshow("detections", frame)