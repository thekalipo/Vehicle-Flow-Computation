from classes.classes import Detector
import cv2
class FasterYoloDetector(Detector):
    def __init__(self, confidence=0.2, nms=0.2):
        self.lastFrame = None
        self.net = cv2.dnn.readNet("yolo-fastest-xl.weights", "yolo-fastest-xl.cfg")
        self.model = cv2.dnn_DetectionModel(self.net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        self.CONFIDENCE_THRESHOLD = confidence
        self.NMS_THRESHOLD = nms

    def findMatches(self, frame):
        classes, scores, boxes = self.model.detect(frame, self.CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)

        matches = []
        for box in boxes:
                matches.append([box,(box[0]+box[2]//2, box[1]+box[3]//2)])
        return matches
