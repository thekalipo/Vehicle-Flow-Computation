from classes.classes import Detector
import cv2
class FrameSubDetector(Detector):
    def __init__(self):
        self.lastFrame = None

    def findMatches(self, frame):
        fgray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.lastFrame is None:
            self.lastFrame = fgray
            return []
        diff = cv2.absdiff(fgray, self.lastFrame)

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

        self.lastFrame = fgray
        return matches
