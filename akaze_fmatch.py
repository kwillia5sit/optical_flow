#!/usr/bin/env python3

from colorsys import rgb_to_hls
import numpy as np
import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import random
from statistics import mean


#define a global counter variable and set as 0. 
i = 0

# parameters for feature detection
feature_params = dict( maxCorners = 200, #maximum number of features
					    qualityLevel = 0.7,  #features with a quality rating of less than this number are rejected
                                            #ex: if best corner is 1500 and 0.01 is qlevel, anything under 15 for score is rejected
				        minDistance = 4,    #minimum distance between the points being picked
				        blockSize = 7)     #average block size for pixel neighborhoods

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (9, 9),              #window size each pyramid level
	    		 maxLevel = 4,                       #number of pyramid levels 
		    	 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 10 iterations, 
                                                                                #criteria count matches the quality level
			    			10, 0.025),
                 flags = 0,                 #Flags determine what the error is calculating.
                 minEigThreshold = 1e-4)     #Threshold for smallest eignevalue minimum that can ount as good data   
                 #zero/not set flags is error from initial position at prevpts
detector = cv2.KAZE_create()

def callback(data):
  #Make the global variables for the named variables
  global prev_frame
  global prev_gray
  global key_points
  global feature_params
  global lk_params
  global i
  global br
  global mask
  global descsi
  #Initialize cv image convertor and AKAZE feature detector
  br = CvBridge()

  #Case 1/2: it's the first frame of the whole series. 
  if i<1:
    # Take first frame of bag message and convert it to openCv image:
    prev_frame = br.imgmsg_to_cv2(data)
    #Take that frame and convert it to a grayscale image:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    key_points, descsi = detector.detectAndCompute(prev_gray, None)
    #Convert key points to coordinate tuples
    key_points = cv2.KeyPoint_convert(key_points)

    #Find the size of the array
    array_size = int(key_points.size)
    my_size = int(array_size/2)
    #Reshape the array to be used by optical flow calculation
    key_points.shape = (my_size, 1, 2)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    i = i+1

  else:
    #Take current frame and make it an openCV grayscale
    cur_frame = br.imgmsg_to_cv2(data)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", cur_frame)
    cv2.waitKey(1)

	# calculate optical flow
    cur_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,
									        cur_gray,
									        key_points, None,
									         **lk_params)
    #Use the resizing array method again on the current points
    cur_size = int(cur_points.size)
    cur_size = int(cur_size/2)
    cur_points.shape = (cur_size, 1, 2)

    #print("initial descs are: ")
    #print(descsi)
    #compute the descriptors for the curret points
    cur_descs = detector.detectAndCompute(cur_gray, None, cur_points)
    #print("current descs are: ")
    #print(cur_descs)

     # Only use good points (had status 1)
    good_cur = cur_points[st == 1]
    good_prev = key_points[st == 1]
  
    # Make a loop to put points into an array
    for s, (cur, prev) in enumerate(zip(good_cur, 
                                       good_prev)):
        #Prepare array to be tuples for the line function
        a, b = cur.ravel()
        c, d = prev.ravel()

        #create a random color
        #r = np.random.randint(0, 255)
        #g = np.random.randint(0, 255)
        #b = np.random.randint(0, 255)
        rgb = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]

        #Print the error
        #print("The error for ", cur, "is:", err[s])
        #"L1 distance between new patch and original patch / pixels in window is error"
        L1 = err[s]*81
        #print("L1 = ", err[s]*9)
        #If error is greater than 3
        if L1>3:
            #draw a line on the mask
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                            rgb, 2)
            frame = cv2.circle(cur_frame, (int(a), int(b)), 5,
                           rgb, -1)
    avg_err= sum(L1>3)/len(L1>3) 
    print("avg_err = ", avg_err)   
    image = cv2.add(frame, mask)
    cv2.imshow('optical flow', image)
    cv2.waitKey(1)
    key_points = cur_points
    prev_gray = cur_gray
    #add 1 to the counter 
    i = i+1
#end of callback loop


def receive_message():
  # Runs once descs1 nce
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('optflowLK_node', anonymous=True)
  # Node is subscribing to the sonar oculus node/image topic
  rospy.Subscriber('/sonar_oculus_node/image', Image, callback)
  rospy.spin()
  cv2.destroyAllWindows()

if __name__=='__main__':
  receive_message()