import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
from math import pi
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import numpy as np
import cv2
from rectAffine import *

## function taken from https://github.com/evanlev/image_rectification ##
def rectifyMetricStrat(imA, nLinePairs, doRotationAfterH = True, doTranslationAfterH = True, doScalingAfterH = True):
    # --------- Supporting functions -----------
    # Plot image and lines
    def replotMetric(imA,limits,lines=[[],[]],x=[[],[]],y=[[],[]]):
        plt.close() # ginput does not allow new points to be plotted
        imshow(imA,cmap='gray')
        axis('image')

        # Plot settings 
        plot_lines = False
        plot_points = True
        
        # Determine how many lines to plot in red, leaving the last in green if the second needs to be picked
        nl1 = len(y[0])
        nl2 = len(y[1])
        if nl1 == nl2:
            nred = nl1
        else:
            nred = nl1 - 1
        if plot_lines:
            for k in range(nred):
                xx,yy = getPlotBoundsLine(limits, lines[0][k])
                plot(xx,yy,'r--')
            if nl1 - nred > 0:
                xx,yy = getPlotBoundsLine(limits, lines[0][nl1-1])
                plot(xx,yy,'g--')
            for l in lines[1]:
                xx,yy = getPlotBoundsLine(limits, l)
                plot(xx,yy,'b--')
        if plot_points:
            # Plot lines: direction 1, all red but the last one green
            for k in range(0,nred):
                plot(x[0][k],y[0][k],'r-')
            if nl1 - nred > 0:
                plot(x[0][nl1-1],y[0][nl1-1],'g-')
            # Plot lines: direction 2
            for k in range(0,len(y[1])):
                plot(x[1][k],y[1][k],'b-')
        axis('off')
        axis('image')
        draw()
    
    ## end of used function ##

    ########## Metric rectification algorithm ##########

    count = 1
    # A = zeros(numConstraints,3);
    A1 = []
    A2 = []
    A = []

    fig2 = plt.figure()
    
    lines = [[],[]]
    x = [[],[]]
    y = [[],[]]

    replotMetric(imA, imA.shape)
    # while (count <= numConstraints)
    for i in range(0, 2*nLinePairs):
        
        ii = i % 2
        plt.suptitle('Select pairs of orthogonal segments')

        if ii == 1:
            plt.suptitle('Click two points intersecting a line perpendicular to the green line')
        else:
            if i == 0:
                plt.suptitle('Click two points intersecting the first of two perpendicular lines')
            else:
                plt.suptitle('Click two points intersecting the first of two perpendicular lines not parallel to any in the first set')
        x1,y1,line = getLine()
        x[ii].append(x1)
        y[ii].append(y1)
        line = line/line[2]
        lines[ii].append(line)

        # each pair of orthogonal lines gives rise to a constraint on s
        # [l(1)*m(1),l(1)*m(2)+l(2)*m(1), l(2)*m(2)]*s = 0
        # store the constraints in a matrix A
        # A(count,:) = [l(1)*m(1),l(1)*m(2)+l(2)*m(1), l(2)*m(2)];
        if ii == 1:
            if i == 1:
                A.append([lines[i-1][0][0]*lines[i][0][0], 
                          lines[i-1][0][0]*lines[i][0][1] + lines[i-1][0][1]*lines[i][0][0],
                          lines[i-1][0][1]*lines[i][0][1]])
            elif i == 3:
                A.append([lines[ii-1][1][0]*lines[ii][1][0],
                          lines[ii-1][1][0]*lines[ii][1][0] + lines[ii-1][1][0]*lines[ii][1][0],
                          lines[ii-1][1][0]*lines[ii][1][0]])

        count = count+1
    
    [_, _, v] = np.linalg.svd(A)
    s = v[:,len(v)-1]
    S = [[s[0], s[1]], [s[1], s[2]]]

    # imDCCP = [[S, np.zeros(2,1)], [np.zeros(1,3)]]
    [U, D, V] = np.linalg.svd(S)
    A = U * np.sqrt(D) * np.transpose(V)
    H = np.identity(3)
    H[0,0] = A[0,0]
    H[0,1] = A[0,1]
    H[1,0] = A[1,0]
    H[1,1] = A[1,1]

    Hrect = np.linalg.inv(H)

    # Scale to keep the output contained just within the image matrix
    if doScalingAfterH:
        Hinv = scaleHToImage(Hrect, imA.shape, False)

    # Do rectification
    imRect = myApplyH(imA, Hinv)

    return imRect, Hinv
