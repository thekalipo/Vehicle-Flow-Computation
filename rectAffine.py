## affine rectification taken from https://github.com/evanlev/image_rectification

import matplotlib.image as mplimg
from pylab import plot, ginput, show, axis, imshow, draw
from math import pi
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg
import numpy as np
import cv2

def getHCorners(H, limits):
    Ny = float(limits[0])
    Nx = float(limits[1])
    # Apply H to corners of the image to determine bounds
    Htr  = np.dot(H, np.array([0.0, Ny, 1.0]))#.flatten(1)) # Top left maps to here
    Hbr  = np.dot(H, np.array([Nx,  Ny, 1.0]))#.flatten(1)) # Bottom right maps to here
    Hbl  = np.dot(H, np.array([Nx, 0.0, 1.0]))#.flatten(1)) # Bottom left maps to here
    Hcor = [Htr,Hbr,Hbl]
    
    # Check if corners in the transformed image map to infinity finite
    finite = True 
    for y in Hcor:
        if y[2] == 0:
            finite = False

    return Hcor, finite

def getPlotBoundsLine(size, l):
    # l = l.flatten('C')
    L = 0
    R = 1
    T = 2
    B = 3
    Nx = size[1]
    Ny = size[0]
    # lines intersecting image edges
    lbd = [[] for x in range(4)]
    lbd[L] = np.array([1.0, 0.0, 0.0])
    lbd[R] = np.array([1.0, 0.0, -float(Nx)])
    lbd[T] = np.array([0.0, 1.0, 0.0])
    lbd[B] = np.array([0.0, 1.0, -float(Ny)])
    I = [np.cross(l, l2) for l2 in lbd]

    # return T/F if intersection point I is in the bounds of the image
    Ied = [] # List of (x,y) where (x,y) is an intersection of the line with the boundary
    for i in [L, R]:
        if I[i][2] != 0:
            In1 = I[i][1] / I[i][2]
            if In1 > 0 and In1 < Ny:
                Ied.append(I[i][0:2]/I[i][2])

    for i in [T, B]:
        if I[i][2] != 0:
            In0 = I[i][0] / I[i][2]
            if In0 > 0 and In0 < Nx:
                Ied.append(I[i][0:2]/I[i][2])

    assert(len(Ied) == 2 or len(Ied) == 0)
    xx = [Ied[x][0] for x in range(0,len(Ied))]
    yy = [Ied[x][1] for x in range(0,len(Ied))]

    return xx,yy

def scaleHToImage(H, limits, anisotropic = False): # TODO: test anisotropic
    assert len(limits) >= 2 # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # If corners in the transformed image are not finite, don't do scaling
    if not finite:
        print("Skipping scaling due to point mapped to infinity")
        return H
        
    # Maximum coordinate that any corner maps to
    k = [max([Hcor[j][i] / Hcor[j][2] for j in range(len(Hcor))])/float(limits[1-i]) for i in range(2)]

    # Scale
    if anisotropic:
        print("Scaling by (%f,%f)\n" % (k[0], k[1]))
        HS = np.array([[1./k[0],0.0,0.0],[0.0,1./k[1],0.0],[0.0,0.0,1.0]])
    else:
        k = max(k)
        print("Scaling by %f\n" % k)
        HS = np.array([[1.0/k,0.0,0.0],[0.0,1.0/k,0.0],[0.0,0.0,1.0]])

    return np.dot(HS, H)

def translateHToPosQuadrant(H, limits):
    assert len(limits) >= 2 # can have color channels
    assert limits[0] > 0 and limits[1] > 0
    assert H.shape[0] == 3 and H.shape[1] == 3

    # Get H * image corners
    Hcor, finite = getHCorners(H, limits)

    # Check if corners map to infinity, if so skip translation
    if not finite:
        print("Corners map to infinity, skipping translation")
        return H

    # Min coordinates of H * image corners
    minc = [min([Hcor[j][i]/Hcor[j][2] for j in range(len(Hcor))]) for i in range(2)]

    # Choose translation
    HT = np.identity(3)
    HT[0,2] = -minc[0]
    HT[1,2] = -minc[1]

    return np.dot(HT, H)

def translateAndScaleHToImage(H, limits, anisotropic = False):
    H = translateHToPosQuadrant(H, limits)
    H = scaleHToImage(H, limits, anisotropic)
    return H

def getLine():
    # get mouse clicks
    pts = []
    while len(pts) == 0: # FIXME
        pts = ginput(n=2)
    pts_h = [[x[0],x[1],1] for x in pts]
    line = np.cross(pts_h[0], pts_h[1]) # line is [p0 p1 1] x [q0 q1 1]

    # return points that were clicked on for plotting
    # x1 = map(lambda x: x[0], pts) # map applies the function passed as 
    # y1 = map(lambda x: x[1], pts) # first parameter to each element of pts

    x1 = pts[0][0]
    y1 = pts[0][1]

    return x1, y1, line

def myApplyH(im, H):
    return cv2.warpPerspective(im, H, (im.shape[1], im.shape[0]))

def rectifyAffineF(im, nLinePairs, doRotationAfterH = True, doTranslationAfterH = True, doScalingAfterH = True):
    # --------- Supporting functions -----------
    # Plot image, lines, vanishing points, vanishing line
    def replotAffine(im,limits,lines=[[],[]],x=[[],[]],y=[[],[]],vPts=[]):
        # -- Settings for this function ---
        plot_lines  = True
        plot_vpts   = True
        plot_points = True
        plot_vline  = True
        # ---------------------------------
        if len(x) != len(y):
            raise Exception("len(x): %d, len(y): %d!" % (len(x), len(y)))
        if len(lines) != len(x):
            raise Exception("len(x): %d, len(lines): %d!" % (len(x), len(lines)))
        if len(vPts) != min(len(x[0]), len(x[1])):
            raise Exception("len(x[0]): %d, len(x[1]): %d, len(vpts): %d!" % (len(x[0]), len(x[1]), len(vPts)))

        plt.close() # ginput does not allow new points to be plotted
        imshow(im,cmap='gray')
        axis('image')
        ax = plt.gca()
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Determine how many lines to plot in red, leaving the last in green if the second needs to be picked
        nl1 = len(y[0])
        nl2 = len(y[1])
        if nl1 == nl2:
            nred = nl1
        else:
            nred = nl1 - 1
        # Plot extension of user-selected lines (dashed)
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

        # Plot user-selected line segments (solid)
        if plot_points:
            # Plot lines: direction 1, all red but the last one green
            for k in range(0,nred):
                plot(x[0][k],y[0][k],'r-')
            if nl1 - nred > 0:
                # print(x[0])
                plot(x[0][nl1-1],y[0][nl1-1],'g-')
            # Plot lines: direction 2
            for k in range(0,len(y[1])):
                plot(x[1][k],y[1][k],'b-')

        # Compute normalized vanishing points for plotting
        vPts_n = [[0,0] for x in vPts]
        vPtInImage = [True for x in vPts]
        for i in range(len(vPts)):
            if vPts[i][2] == 0:
                vPtInImage[i] = False
            else:
                vPts_n[i][0] = vPts[i][0] / vPts[i][2]
                vPts_n[i][1] = vPts[i][1] / vPts[i][2]
                vPtInImage[i] = vPts_n[i][0] < limits[0] and vPts_n[i][0] > 0 and vPts_n[i][1] < limits[1] and vPts_n[i][1] > 0

        # Plot vanishing points 
        if plot_vpts:
            for i in range(len(vPts_n)):
                if vPtInImage[i]:
                    plot(vPts_n[i][0], vPts_n[i][1], 'yo')
        # Plot vanishing line
        if plot_vline:
            if len(vPts) == 2 and vPtInImage[0] and vPtInImage[1]:
                vLine = np.linalg.cross(vPts[0], vPts[1])
                xx,yy = getPlotBoundsLine(limits, vLine)
                plot(xx,yy,'y-')
                #plot([vPts_n[0][0],vPts_n[1][0]], [vPts_n[0][1],vPts_n[1][1]], 'y-')

        # Limit axes to the image
        plt.xlim([xmin,xmax])
        plt.ylim([ymin,ymax])


    # -------------- START -----------------

    # Create figure
    fig2 = plt.figure()
    replotAffine(im,im.shape)

    # Lines
    lines = [[],[]]
    x = [[],[]]
    y = [[],[]]
    vPts = []
        
    # Get line pairs interactively
    for i in range(0,2*nLinePairs):
        ii = i % 2
        if ii == 1:
            plt.suptitle('Click two points intersecting a line parallel to the green line')
        else:
            if i == 0:
                plt.suptitle('Click two points intersecting the first of two parallel lines')
            else:
                plt.suptitle('Click two points intersecting the first of two parallel lines not parallel to the first set')
        x1,y1,line = getLine()
        x[ii].append(x1)
        y[ii].append(y1)
        lines[ii].append(line)
        if ii == 1:
            nlp = len(lines[0])
            vPt = np.cross(lines[0][nlp-1], lines[1][nlp-1])
            if vPt[2] != 0.:
                vPt[0] = vPt[0] / vPt[2]
                vPt[1] = vPt[1] / vPt[2]
                vPt[2] = vPt[2] / vPt[2]
            vPts.append(vPt)
        # re-plot figure
        replotAffine(im,im.shape,lines,x,y,vPts)

    vLine = np.cross(vPts[0], vPts[1])

    H = np.identity(3)
    H[2,0] = vLine[0] / vLine[2]
    H[2,1] = vLine[1] / vLine[2]

    # Scale to keep the output contained just within the image matrix
    # if doScalingAfterH:
    #     H = scaleHToImage(H, im.shape, False)

    # Apply H to do affine rectification
    imRect = myApplyH(im, H)

    plt.close(fig2)
    return imRect, H
