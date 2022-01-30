import cv2
import numpy as np

c = cv2.VideoCapture("video.mp4")
_,f = c.read()

avg1 = np.float32(f)
avg2 = np.float32(f)

while(1):
    _,f = c.read()
	
    cv2.accumulateWeighted(f,avg1,0.1)
    cv2.accumulateWeighted(f,avg2,0.01)
	
    res1 = cv2.convertScaleAbs(avg1)
    res2 = cv2.convertScaleAbs(avg2)

    cv2.imshow('img',f)
    cv2.imshow('avg1',res1)
    cv2.imshow('avg2',res2)
    k = cv2.waitKey(20)

    if k == 27:
        break

res1 = cv2.convertScaleAbs(avg1)
res2 = cv2.convertScaleAbs(avg2)
cv2.imwrite("avg1.png",res1)
cv2.imwrite("avg2.png",res2)

cv2.destroyAllWindows()
c.release()

"""
import cv2
import numpy as np
cap = cv2.VideoCapture('video.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")
frames = []
ret,frame1 = cap.read()
ret,frame2 = cap.read()
avg1 = np.float32(frame1)
avg1 += np.float32()
cv2.imshow('Frame1',frame1)
cv2.imshow('Frame2',frame2)
cv2.waitKey()
mean = np.mean([frame1,frame2])
mean = mean.astype(np.uint8)
cv2.imshow('Frame mean',mean)
cv2.waitKey()
"""
"""
for i in range (5):
    _,f = cap.read()
    frame2 += f
    frames.append(f)

frame = np.median(frames)
frame2 = frame2 / (len(frames)+1)
cv2.imshow('Frame',frame)
cv2.imshow('Frame',frame2)
cv2.waitKey()
"""
"""
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:

    # Display the resulting frame
    cv2.imshow('Frame',frame)

    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
"""
"""
import numpy as np
import cv2
#from skimage import data, filters

# Open Video
cap = cv2.VideoCapture('video.mp4')

# Randomly select 25 frames
frameIds = #cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

# Display median frame
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
"""