import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

import os,sys,warnings
warnings.filterwarnings('ignore')

import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

def do_canny(frame):
    
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)    
    blur = cv.GaussianBlur(gray, (13, 13), 0)    
    canny = cv.Canny(blur,50,150)
    return canny

one = cv.imread(r"india.jpg")
one = cv.cvtColor(one,cv.COLOR_BGR2RGB)
plt.imshow(one)

two = do_canny(one)
plt.imshow(two)
plt.savefig("Cannysample.jpg",dpi=200)

def do_segment(frame):
    
    height = frame.shape[0]
    width = frame.shape[1]
    
    poly = np.array([
        [(0,height),(width,height),(int(frame.shape[1]/2),int(frame.shape[0]/3))] 
    ])
    
    mask = np.zeros_like(frame)
    cv.fillPoly(mask,poly,255) 
    
   
    segment = cv.bitwise_and(frame,mask)
    
    return segment

plt.imshow(one)
plt.title("Original Image",fontsize =18)

plt.imshow(do_segment(do_canny(one)))
plt.savefig("Cannysegment.jpg",dpi=200)


def calculate_lines(frame,lines):
    # Empty arrays to store the co-ordinates of left and right lines
    left = []
    right = []
    if(lines is None):
        return (np.array([0,0,0,0]),np.array([0,0,0,0]))
    for line in lines:
        #loop through the lines found.
        x1,y1,x2,y2 = line.reshape(4)
        # fit a linear polynomial to the x and y coordinates 
        # and returns a vector of coefficients which describe 
        # the slope and the intercept.
        parameters = np.polyfit((x1,x2),(y1,y2),1)
        slope = parameters[0]
        y_intercept = parameters[1]
         # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis = 0)
    right_avg = np.average(right, axis = 0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    if(left_avg!=[]):
        left_line = calculate_coordinates(frame, left_avg)
    else:
        left_line = np.array([0,0,0,0])
    if(right_avg!=[]):
        right_line = calculate_coordinates(frame, right_avg)
    else:
        right_line = np.array([0,0,0,0])
    return np.array([left_line, right_line])

def calculate_coordinates(frame, parameters):
    try:
        slope, intercept = parameters
    except:
        slope = 0.001
        intercept = 0
    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = np.int64(frame.shape[0])
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = np.int64(y1 - 160)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = np.int64( (y1-intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = np.int64((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    # Creates an image filled with zero intensities with the same dimensions as the frame
    lines_visualize = np.zeros_like(frame)
    # Checks if any lines are detected
    if lines is None:
        return lines_visualize
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            # Draws lines between two coordinates with green color and 5 thickness
            try:
                cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
                diff = x2-x1
                print(diff)
            except:
                pass
    return lines_visualize



def rescale_frame(frame,percent=75):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width,height)
    return cv.resize(frame,dim,interpolation = cv.INTER_AREA)


cap = cv.VideoCapture(r"sample1.mp4")
while (cap.isOpened()):
    # ret = a boolean return value from getting the frame,
    # frame = teh current frame being projected in video
    
    # frame = cv.imread('Untitled.png')
    ret, frame = cap.read()

    try:
        frame = rescale_frame(frame,percent = 75)
    except:
        print(frame)
        break
    canny = do_canny(frame)
    # First, visualize the frame to figure out the three coordinates defining the triangular mask
    #plt.imshow(frame)
    # cv.imshow("Original",frame)
    #cv.namedWindow("Original",cv.WINDOW_NORMAL)
    #cv.resizeWindow("Original",500,300)
    # cv.imshow("Canny filtered",canny)
    #cv.windowSize("Original",400,400)
    # making the segment constrained so that detection is
    # confined.
    segment = do_segment(canny)
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength = 80, maxLineGap = 50)
    # cv.imshow("Segmented canny",segment)
    # frames read at 10 Millsecond intervals.
    # Averages multiple detected lines from hough into one line for left border of lane and one line for right border of lane
    lines = calculate_lines(frame, hough)
    # Visualizes the lines
    lines_visualize = visualize_lines(frame, lines)
    #cv.imshow("hough", lines_visualize)
    # Overlays lines on frame by taking their weighted sums and adding an arbitrary scalar value of 1 as the gamma argument
    output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    # Opens a new window and displays the output frame
    cv.imshow("output", output)
    # cv.imwrite("output.jpg",output)
    # break
    cv.VideoWriter()
    
    if cv.waitKey(15) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
# releasing all the resources.
