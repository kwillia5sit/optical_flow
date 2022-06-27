#!/usr/bin/env python3

import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

def callback(data):

    #Make the global variables for the named variables
    global prev_frame
    global prev_gray
    global prev_features
    global feature_params
    global lk_params

    #define a global counter variable and set as 0.
    global i 
    i = 0
    #define CvBridge function
    br=CvBridge()

    # parameters for feature detection
    feature_params = dict( maxCorners = 100, #maximum number of features
					    qualityLevel = 0.3,  #features with a quality rating of less than this number are rejected
                                        #ex: if best corner is 1500 and 0.01 is qlevel, anything under 15 for score is rejected
				        minDistance = 7,    #minimum distance between the points being picked
				        blockSize = 7 )     #average block size for pixel neighborhoods

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize = (15, 15),              #window size each pyramid level
	    			maxLevel = 2,                       #number of pyramid levels 
		    		criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 10 iterations, 
                                                                                        #criteria count matches the quality level
			    				10, 0.03))

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

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

    else:
	    #Take current frame and make it an openCV grayscale
        cur_frame = br.imgmsg_to_cv2(data)
        cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)

	    # calculate optical flow
        cur_features = cv2.calcOpticalFlowPyrLK(prev_gray,
										        cur_gray,
										        prev_features, None,
										         **lk_params)
        prev_r = cv2.calcOpticalFlowPyrLK(cur_gray, prev_gray, cur_features, None, **lk_params)
        #Measure change between original prev_point and prev_r
        d = abs(prev_features-prev_r).reshape(-1, 2).max(-1)

	    # Select good points
        #d < 1, the point is good
        if d<1:
            mask = cv2.line(mask, cur_features.ravel(), prev_features.ravel(), color.tolist(), 2)
            frame = cv2.circle(frame, cur_features, 5, color.tolist(), -1)
            img = cv2.add(frame, mask)

            #Show the tracks
            cv2.imshow('frame', img)

            #Update previous frame and features
            prev_gray = cur_gray
            prev_features = (prev_features-prev_r).reshape(-1, 1, 2)
        cv2.waitKey(1)


#initialize subscriber node
rospy.init_node('optflow_node', anonymous=True)
rospy.Subscriber('/sonar_oculus_node/M1200d/image', Image, callback)  
#Run the bag
rospy.spin()

cv2.destroyAllWindows()