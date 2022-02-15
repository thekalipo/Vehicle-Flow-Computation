# Vehicle-Flow-Computation
Vehicle Flow Computation: 

Politecnico di Milano Image Analysis and Computer Vision Project

- Glen Pouliquen
- Andres Gonzalez Paul Rivera

----

### Sections:

1. [Project description](#subject-vehicle-flow-computation)

2. [Proposed solution](#proposed-solution)

    - [Object detection](#object-detection)
    - [Object tracking](#object-tracking)
    - [Vanishing point calculation](#vanishing-point-calculation)
    - [Cross ratio](#cross-ratio)
    - [Speed detection](#speed-detection)

3. [Dependencies](#dependencies)

4. [How to use the code](#how-to-use)

5. [Use on different video](#use-on-different-video)

6. [Modularity](#modularity)

7. [Resources](#resources)

----

## Subject: Vehicle Flow Computation
Visual analysis of/from moving vehicles

Topic (ii): Vehicle Flow Computation

Develop a program that has the following:

INPUT:
* Images or videos: A video taken from a fixed camera. Camera is placed at a certain distance (height) over the road plane
* Scene: A road. One or two visible fixed natural markers on the road (e.g., poles,  road signs, or pederstrian crossings). Vehicles driving on the road.
* Data on camera and scene: Both the position of the camera and its calibration parameters are UNKNOWN. The distance between the two natural markers on the road is KNOWN.

OUTPUT:
* Detect individual vehicles driving on the road
* Compute number of vehicles that cross one of the two road markers per each fifteen second and the average of their speed in the crossing instant.

----

## Proposed Solution

Our approach to develop the mentioned program is based on different steps in order to get the final outputs. The steps are the following and will be explained in detail.

1. Object detection
2. Object tracking
3. Vanishing point calculation
4. Cross ratio
5. Speed detection

### **Object detection**

For the object detection of this project we selected two different algorithms. 

First we use a frame by frame differentiation and apply a threshold to obtain sharper results. Then we dilate the image in order to avoid holes on the moving objects. Finally we get contours on the moving objects.

The second one is the YOLO (You Only Look Once) algorithm which is a realtime object detection algorithm that uses a single neural network on the image frame, it divides it into regions and predicts bounding boxes and probabilities for each region, which are weighted by their probabilities. More preciselly we use [Yolo Fastest](https://github.com/dog-qiuqiu/Yolo-Fastest).

### **Object tracking**

Once we detected objects we can track them. We use the [SORT](https://github.com/abewley/sort) (Simple Online Realtime Tracking) algorithm which is a multi-object tracking system. This algorithm extracts features of the detected objects, it then uses a Kalman filter to predict the location, calculates the features similarities and the location distance between different frames. Computes the intersection-over-union (IOU) between the detection and the predicted candidate, and finally it updates the motion state by the Kalman filter and the motion prediction model.

### **Vanishing point calculation**

The vanishing point on an image is the point were, parallel lines in the real world, intersect. This point is calculated by the cross product of the lines. It is the projection of the point at infinity of parallel lines on the real world.

For our project we calculate the vanishing point by calculating the direction vectors of the cars moving along a lane, since we know that lanes are parallel in real life, the image lines will intersect on the real point at infinity. The more direction vectors of the cars and the more cars we have, the more accurate the vanishing point will be. 

NOTE: direction vectors should be obtained from the first detected point of a car and the last, so that the length is the longest, thus get more accurate results.

### **Cross Ratio**

The cross ratio is a ratio of ratios, it is obtained by a 4-tuple of collinear points that project to a real distance between points.
We compute it by using the vanishing point (point at infinity), two markers that are already known, and the position of the car on the same line as the other points.
For cars on different lanes we compute the marker points by the intersection with the line from the car to the vanishing point.

With the cross ratio we can obtain the distance the car has moved which is used for the speed detection.

### **Speed detection**

In order to calculate the speed of the vehicles, we use the video parameters and the computed positions of the vehicles by the cross ratio in order to get real distances, thus real speed estimation.

We know that: speed = distance / time

Since we already obtained the distance of the vehicles, we need to compute the time, which is obtained by using the frames per second (fps) of the video. For each vehicle at a specific instant we obtain the diference of frames between the current one and the frame used to calculate the distance and divide it by the fps of the video.

The average speed is also calculated using the time the vehicle puts to go from one marker to another.

Finally the speed is multiplied by 3.6 on order to get km/h.

Log files are created every 15 secs containing the average speed between the two lines of the counted vehicles in the last 15 secs.

----

## Dependencies

To install required dependencies run:

```
$ pip install -r requirements.txt
```


## How to use

To run the program run:

```
$ cd path/to/program
$ python main.py <number of video to use>
```

The number of video goes from 0 to 2 in our specific case, but you can add new videos in the first condition of `main.py`.

## Use on a different video
For new videos you'll need to specify the name of the video, whose file needs to be on the same folder, and define two lines that are parallel in the real word and add the distance in meter between these two.

## Modularity

The program is structured in a way that is easy to add or modify any part of it. If you'd like to use a different detector or a different tracker model, you can simply extend one of the classes of the `classes.py` file and make use of the modularity.
All classes are independent, a tracker can work con different detectors and vice-versa.

----

## Resources
### Sources
https://ieeexplore.ieee.org/document/9469561

https://github.com/andrewssobral/simple_vehicle_counting

http://www.insecam.org/en/

https://jorgestutorials.com/pycvtraffic.html

https://github.com/topics/vehicle-counting

https://github.com/KEDIARAHUL135/VanishingPoint/blob/master/main.py

https://github.com/evanlev/image_rectification

https://ieeexplore.ieee.org/document/9469561

https://github.com/andrewssobral/simple_vehicle_counting

https://www.researchgate.net/figure/Comparison-of-existing-state-of-the-art-methods_tbl1_341259906

http://www.insecam.org/en/

https://jorgestutorials.com/pycvtraffic.html

https://github.com/topics/vehicle-counting

https://learnopencv.com/simple-background-estimation-in-videos-using-opencv-c-python/


https://research.ijcaonline.org/volume102/number7/pxc3898647.pdf

https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/

https://learnopencv.com/object-tracking-using-opencv-cpp-python/

https://github.com/nicholaskajoh/ivy

https://stackoverflow.com/questions/4260594/image-rectification

https://www.bogotobogo.com/cplusplus/files/OReilly%20Learning%20OpenCV.pdf

https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html

https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html

https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/

https://nanonets.com/blog/object-tracking-deepsort/

https://stackoverflow.com/questions/36254452/counting-cars-opencv-python-issue/36274515#36274515
