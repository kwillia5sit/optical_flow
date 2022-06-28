#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

#define a global counter variable and set as 0. 
i = 0

# parameters for feature detection
feature_params = dict( maxCorners = 200, #maximum number of features
					    qualityLevel = 0.5,  #features with a quality rating of less than this number are rejected
                                    #ex: if best corner is 1500 and 0.01 is qlevel, anything under 15 for score is rejected
				        minDistance = 2,    #minimum distance between the points being picked
				        blockSize = 7)     #average block size for pixel neighborhoods

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (4, 4),              #window size each pyramid level
	    			maxLevel = 3,                       #number of pyramid levels 
		    		criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 10 iterations, 
                                                                                        #criteria count matches the quality level
			    				10, 0.03))
# Create some random colors (for drawing lines later)
color = np.random.randint(0, 255, (100, 3))

def callback(data):
  #Make the global variables for the named variables
  global prev_frame
  global prev_gray
  global prev_features
  global feature_params
  global lk_params
  global i
  global br
  global mask
  br = CvBridge()


  #Case 1/2: it's the first frame of the whole series. 
  if i<1:
    # Take first frame of bag message and convert it to openCv image:
    prev_frame = br.imgmsg_to_cv2(data)
    #Take that frame and convert it to a grayscale image:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    #Take that grayscale image and find the significant features
    prev_features = cv2.goodFeaturesToTrack(prev_gray, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    i = i+1

  else:
	  #Take current frame and make it an openCV grayscale
    cur_frame = br.imgmsg_to_cv2(data)
    cv2.imshow("image", cur_frame)
    cv2.waitKey(1)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
	  # calculate optical flow
    cur_features, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,
									        cur_gray,
									        prev_features, None,
									         **lk_params)
    print("Have optical flow")

    # Only use good points (had status 1)
    good_cur = cur_features[st == 1]
    good_prev = prev_features[st == 1]
  
    # Make a loop to put points into an array
    for s, (cur, prev) in enumerate(zip(good_cur, 
                                       good_prev)):
      #Prepare array to be tuples for the line function
      a, b = cur.ravel()
      c, d = prev.ravel()
      print(cur)
      print("the cur type is", type(cur))
      #The next set of comments are all the other commands I tried
      #a, b = np.stack(int(prev))
      #c, d = np.stack(int(cur))
      #a, b=np.ravel(prev, tuple)
      #c, d=np.ravel(cur, tuple)
      #print("the c, d type is ", type(c, d))
      #print(c, d)
      #print(type(c, d))
      #prev_tuple = tuple(map(tuple, prev[s]))
      #cur_tuple = tuple(map(tuple, cur[s]))
      #print("The cur_tuple types is", type(cur_tuple[s]))

      #draw a line on the mask
      mask = cv2.line(mask,(a, b), (c, d),
                        color[s].tolist(), 2)
      frame = cv2.circle(cur_frame, prev, cur, 5,
                           color[s].tolist(), -1)
          
    #img = cv2.add(frame, mask)
  
    cv2.imshow('frame', frame)

        #update previous frame and features
    prev_gray = cur_gray
    prev_features = cur_features
    #add 1 to the counter 
    i = i+1
    cv2.waitKey(1)
#end of callback loop
    

def receive_message():
  # Runs once
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('optflowLK_node', anonymous=True)
  # Node is subscribing to the video_frames topic
  rospy.Subscriber('/sonar_oculus_node/M1200d/image', Image, callback)
  rospy.spin()
  cv2.destroyAllWindows()

if __name__=='__main__':
  receive_message()
