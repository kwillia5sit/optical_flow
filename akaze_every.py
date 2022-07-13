#!/usr/bin/env python3
#include <opencv2/features2d.hpp>

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
# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (9, 9),              #window size each pyramid level
	    		 maxLevel = 4,                       #number of pyramid levels 
		    	 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 10 iterations, 
                                                                                #criteria count matches the quality level
			    			4, 0.025),
                 flags = 0,                 #Flags determine what the error is calculating.
                 minEigThreshold = 1e-3)     #Threshold for smallest eignevalue minimum that can ount as good data   
                 #zero/not set flags is error from initial position at prevpts
#parameters for FLANN Matcher 
index_params= dict(algorithm = 6, #FLANN_INDEX_LSH = 6
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=100)

#kaze feature detector
detector = cv2.KAZE_create()
#create object for brute-force feature matcher
matcher = cv2.BFMatcher()


#Create array of random colors for drawing purposes
no_of_colors = 500
#create empty array with 3 rows
color_array = np.empty((0,3), int)
for n in range(no_of_colors):
  #Generate random color 
  color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
  #Add color to the array
  color_array = np.append(color_array, np.array([color]), axis=0)

def callback(data):
  #Make the global variables for the named variables
  global prev_frame
  global prev_gray
  global key_points
  global lk_params
  global i
  global br
  global mask
  global desi
  global raw_key_points
  global my_size
  #Initialize cv image convertor and AKAZE feature detector
  br = CvBridge()

  #Case 1/2: it's the first frame of the whole series. 
  if i<1:
    # Take first frame of bag message and convert it to openCv image:
    prev_frame = br.imgmsg_to_cv2(data)
    #Take that frame and convert it to a grayscale image:
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    #find the kaze key points
    raw_key_points, desi = detector.detectAndCompute(prev_gray, None)

    #Convert key points to coordinate tuples
    key_points = cv2.KeyPoint_convert(raw_key_points)
    #Find the size of the array
    array_size = int(key_points.size)
    my_size = int(array_size/2)
    #Reshape the array to be used by optical flow calculation
    key_points.shape = (my_size, 1, 2)
    print("keypoints length is ", my_size)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_frame)
    #add one to counter
    i = i+1

  else:
    #Take current frame and make it an openCV grayscale
    cur_frame = br.imgmsg_to_cv2(data)
    cur_gray = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("image", cur_frame)
    cv2.waitKey(0)

    #Find kaze key points in current frame 
    raw_cur_key_points, descur = detector.detectAndCompute(cur_gray, None)
    #Convert current key points to coordinate tuples
    cur_key_points = cv2.KeyPoint_convert(raw_cur_key_points)
    #Find the size of the array
    array_size = int(cur_key_points.size)
    cur_size = int(array_size/2)
    #Reshape the array to be used by optical flow calculation
    cur_key_points.shape = (cur_size, 1, 2)

    #Find matches between the previous keypoints and the current keypoints
    matches = matcher.knnMatch(descur,desi, k = 1)
    output = cv2.drawMatchesKnn(cur_gray, raw_cur_key_points,
    prev_gray, raw_key_points, matches[:20],None)
    cv2.imshow("keypoint matches", output)
    cv2.waitKey(0)

    cur_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,
									        cur_gray,
									        key_points, None,
									         **lk_params)
    #Use the resizing array method again on the current points
    cur_size = int(cur_points.size)
    cur_size = int(cur_size/2)
    cur_points.shape = (cur_size, 1, 2)

    # Only use good points (had status 1)
    good_cur = cur_points[st == 1]
    good_prev = key_points[st == 1]

    # Make a loop to put points into an array
    for s, (cur, prev) in enumerate(zip(good_cur, 
                                       good_prev)):
        #Prepare array to be tuples for the line function
        a, b = cur.ravel()
        c, d = prev.ravel()

        #Print the error
        #print("The error for ", cur, "is:", err[s])
        #"L1 distance between new patch and original patch / pixels in window is error"
        L1 = err[s]*9
        #print("L1 = ", err[s]*9)
        #If error is greater than 3 and less than 500:
        if L1>3 and L1 < 500:
            #Pick color number s from the array and turn its numbers into a list
            rand_color = color_array[s].tolist()
            #draw a line on the mask
            #mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                            #rand_color, 2)
            flow_cur = cv2.circle(cur_frame, (int(a), int(b)), 5,
                           [0, 255, 255], -1)
            
            
    #avg_err= sum(L1)/s
    #print("avg_err = ", avg_err)   
    #image = cv2.add(flow_cur, dot_cur_key)
    cv2.imshow('optical flow', flow_cur)
    cv2.waitKey(0)
    key_points = cur_key_points
    prev_gray = cur_gray
    desi = descur
    #add 1 to the counter 
    i = i+1
#end of callback loop

def receive_message():
  # Runs once
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random numbers are added to the end of the name. 
  rospy.init_node('optflowLK_node', anonymous=True)
  # Node is subscribing to the sonar oculus node/image topic
  rospy.Subscriber('/sonar_oculus_node/image', Image, callback)
  rospy.spin()
  cv2.destroyAllWindows()

if __name__=='__main__':
  receive_message()