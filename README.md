# Vehicle-Flow-Computation
vehicle flow computation, Politecnico Milano Image Analysis and Computer Vision project

## subject Vehicle Flow Computation
Visual analysis of/from moving vehicles

Topic (ii): Vehicle Flow Computation
Develop a program that uses as

INPUT: 
* Images or videos: A video taken from a fixed camera. Camera is placed at a certain distance (height) over the road plane
* Scene: A road. One or two visible fixed natural markers on the road (e.g., poles,  road signs, or pederstrian crossings). Vehicles driving on the road.
* Data on camera and scene: Both the position of the camera and its calibration parameters are UNKNOWN. The distance between the two natural markers on the road is KNOWN.

 and computes the following

OUTPUT: 
* detect individual vehicles driving on the road
* compute number of vehicles that cross one of the two road markers per each fifteen second and the average of their speed in the crossing instant.

possibly exploiting the following

HINTS: 
* first compute the vanishing point of the road direction (by tracking vehicle features and intersecting their imaged motion directions)
* apply the cross ratio invariance to compute the coordinate of any vehicle along the road direction in correspondence to each frame. 
