import cv2
import numpy as np
from matplotlib import pyplot as plt
  
  
img = cv2.imread('avg1.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols, ch = img.shape
  
pts1 = np.float32([[321, 213], 
                   [348, 122],
                   [424, 221]])

[
[425,  22], [567, 300]
[247, 163], [295,  95]
[415, 172], [420, 220]
[321, 212], [334, 167]
]
  
pts2 = np.float32([[50, 50], 
                   [200, 50],
                   [50, 200]])


np.float32([[10, 100],
            [200, 50], 
            [100, 250]])
  
M = cv2.getAffineTransform(pts1, pts2)

dst = cv2.warpAffine(img, M, (cols, rows))
      
cv2.imshow('image', img)
cv2.imshow('dst', dst)
cv2.waitKey()          
cv2.destroyAllWindows()
"""
from random import randint
from sys import maxsize
import cv2
import numpy as np



img = cv2.imread('avg2.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

kernel_size = 5
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

low_threshold = 100
high_threshold = 150
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 50  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 30  # minimum number of pixels making up a line
max_line_gap = 5  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

lines = [lines[0], lines[13], lines[20], lines[25]]
maxline = (0,0,0,0)
maxsize = 0
i=0
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(line_image,str(i), (x1,y1), 0, 0.4, 255)
        #cv2.imshow("img",cv2.addWeighted(img, 0.8, line_image, 1, 0))
        #cv2.waitKey()
    i += 1

# Draw the lines on the  image
lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

cv2.imshow("res",lines_edges)

cv2.waitKey(0)

cv2.imwrite("lines.png", lines_edges)
print(lines)
"""