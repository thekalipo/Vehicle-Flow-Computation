from logging import critical
import cv2
import math
import time
import numpy as np

from rectAffine import rectifyAffineF, translateAndScaleHToImage, myApplyH
from rectMetStrat import rectifyMetricStrat

REJECT_DEGREE_TH = 4.0

def roiImage(frame, vectors):
    mask = np.zeros_like(frame)
    if len(frame.shape) > 2:
        ch_cnt = frame.shape[2]
        mask_color = (255,) * ch_cnt 
    else:
        mask_color = 255
    cv2.fillPoly(mask, vectors, mask_color)
    masked = cv2.bitwise_and(frame, mask)
    return masked

def filterLines(lines):
    finalLines = []
    fLines = []
    
    for line in lines:
        [[x1, y1, x2, y2]] = line

        # Calculating equation of the line: y = mx + c
        if x1 != x2:
            m = (y2 - y1) / (x2 - x1)
        else:
            m = 100000000
        c = y2 - m*x2
        # theta will contain values between -90 -> +90. 
        theta = math.degrees(math.atan(m))

        # Rejecting lines of slope near to 0 degree or 90 degree and storing others
        if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
            l = math.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )    # length of the line
            finalLines.append([x1, y1, x2, y2, m, c, l])
            fLines.append([x1, y1, x2, y2, l])

    
    # Removing extra lines 
    # (we might get many lines, so we are going to take only longest 15 lines 
    # for further computation because more than this number of lines will only 
    # contribute towards slowing down of our algo.)
    # if len(finalLines) > 1:
    #     finalLines = sorted(finalLines, key=lambda x: x[-1], reverse=True)
    #     finalLines = finalLines[:15]
    
    return finalLines, fLines

def linesCross(lines):
    crossLines = []
    # for l in range(len(lines)):
    for line in lines:
        a = [line[0][0], line[0][1], 1]
        b = [line[0][2], line[0][3], 1]
        line = np.cross(a, b)
        crossLines.append(line)
    
    return crossLines

def getLines(frame, roi):    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    grayBlur = cv2.GaussianBlur(gray, (5, 5), 1)
    
    edges = cv2.Canny(grayBlur, 100, 150)

    cropped_frame = roiImage(edges, np.array([roi], np.int32))
    cv2.imshow('crop', cropped_frame)
    
    # cropped_frames = roiImage(frame, np.array([roi], np.int32))
    # cv2.imshow('crop', cropped_frames)

    # lines = cv2.HoughLinesP(cropped_frame, 1, np.pi / 180, 20, 15, 10)
    if roi_sel == 0:
        lines = cv2.HoughLinesP(cropped_frame, 6, np.pi / 180, 20, np.array([]), 40, 25)
    elif roi_sel == 1:
        lines = cv2.HoughLinesP(cropped_frame, 6, np.pi / 180, 20, np.array([]), 80, 10)

    if lines is None:
        print('no lines found')
    
    cLines = linesCross(lines)
    f_lines, l = filterLines(lines)
    
    return f_lines, l, cLines

def vanishingPointCross(lines):
    for i in range(len(lines)):
        if i+1 < len(lines):
            v = np.cross(lines[i], lines[i+1])
            v = v/v[2]
    return v

def vanishingPoint(lines):
    vPoint = None
    minError = 100000000000

    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            m1, c1 = lines[i][4], lines[i][5]
            m2, c2 = lines[j][4], lines[j][5]

            if m1 != m2:
                x0 = (c1 - c2) / (m2 - m1)
                y0 = m1 * x0 + c1

                err = 0
                for k in range(len(lines)):
                    m, c = lines[k][4], lines[k][5]
                    m_ = (-1 / m)
                    c_ = y0 - m_ * x0

                    x_ = (c - c_) / (m_ - m)
                    y_ = m_ * x_ + c_

                    l = math.sqrt((y_ - y0)**2 + (x_ - x0)**2)

                    err += l**2

                err = math.sqrt(err)
                if minError > err:
                    minError = err
                    vPoint = [x0, y0]
    
    if vPoint is None:
        print('no vanishing point found')

    return vPoint

def main():

    vid = cv2.VideoCapture('video.mp4')

    while True:
        ret, frame = vid.read()

        current_fps = vid.get(cv2.CAP_PROP_POS_FRAMES)

        height = frame.shape[0]
        width = frame.shape[1]
        # print(height, width)

        roi = [
            [
                (50, height),
                (360, 0),
                (460, 0),
                (width, height),
            ],
            [
                (0, height),
                (0, 1700),
                (width/2 - 200, 300),
                (width, 1500),
                (width, height),
            ]
        ]

        global roi_sel
        roi_sel = 0

        if int(current_fps) == 1:
            # vPoint = vanishingPoint(lines)
            
            lines, fLines, cLines = getLines(frame, roi[roi_sel])
            vPointCross = vanishingPointCross(cLines)
            
            # print(vPointCross)

            imA, HA = rectifyAffineF(frame, 2)
            imM, HM = rectifyMetricStrat(imA, 2)
            H = translateAndScaleHToImage(np.dot(HM,HA), imA.shape)

        # cv2.imshow('affine', myApplyH(frame, HA))
        # cv2.imshow('metric', myApplyH(frame, HM))
        im = myApplyH(frame, H)
        cv2.imshow('im', im)

        if roi_sel == 0:
            x = 1
            for line in lines:
                if x == 5 or x == 7:
                    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 2)
                    cv2.circle(frame, (line[2], line[3]), 10, (0, 0, 255), -1)
                elif x == 1 or x == 2 or x == 3 or x == 4 or x == 6 or x == 8 or x == 9:
                    cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
                x = x + 1
        elif roi_sel == 1:
            for line in lines:
                cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        # distance_x = np.linalg.norm(lines[7][2] - lines[5][2])
        # distance_y = np.linalg.norm(lines[7][3] - lines[5][3])
        # dist = math.sqrt(distance_x**2 + distance_y**2)
        # print('dist', dist)

        distance = math.sqrt(((lines[7][2] - lines[5][2])**2) + ((lines[7][3] - lines[5][3])**2))

        cv2.circle(frame, (int(vPointCross[0]), int(vPointCross[1])), 20, (0, 0, 255), -1)

        cv2.imshow('frame', frame)

        key = cv2.waitKey(0)

        if key == ord('q') or key == 27:
            break
        
        time.sleep( 1.0 / vid.get(cv2.CAP_PROP_FPS) )
    
    print('Closing video')
    vid.release()
    cv2.destroyAllWindows()
    print('Finished')

if __name__ == '__main__':
    main()
