import cv2
import time

cap = cv2.VideoCapture('video.mp4')  #Path to footage

# 
from object_detection import ObjectDetection
from deep_sort.deep_sort import Deep
od = ObjectDetection("yolo-fastest-xl.weights", "yolo-fastest-xl.cfg")
od.load_class_names("coco.names")
od.load_detection_model(image_size=640,
                        nms_threshold=0.4,
                        confThreshold=0.3)

deep = Deep(max_distance=0.7,
            nms_max_overlap=1,
            n_init=3,
            max_age=15,
            max_iou_distance=0.7)

tracker = deep.sort_tracker()

while True:
    ret, img = cap.read()
    
    width  = cap.get(3)
    height = cap.get(4)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.8, 2)

    # from video Sergio
    (class_ids, scores, boxes) = od.detect(img)

    features = deep.encoder(frame, boxes)
    detections = deep.Detection(boxes, scores, class_id, features)

    tracker.predict()
    (class_ids, object_ids, boxes) = tracker.update(detections)

    for class_id, object_id, box in zip(class_ids, object_ids, boxes):
        (x, y, x2, y2) = box
        class_name = od.classes[class_id]
        color = od.colors[class_id]
        cx = int((x + x2) / 2)
        cy = int((y + y2) / 2)

        if class_name in ["car", "truck"]:
            cv2.rectangle(img, (x, y), (x2, y2), color, 2)
            cv2.rectanlge(img, (x, y), (x + 100, y - 30), color, -1)
            cv2.putText(img, class_name + " " + str(object_id), (x, y - 10), 0, 0.75, (255, 255, 255), 2)

            # result = cv2.pointPolygonTest(np.array(area_1))

    cv2.imshow('img', img) #Shows the frame

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()