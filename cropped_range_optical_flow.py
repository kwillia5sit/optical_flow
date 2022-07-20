#Can input an array of points for a polygon on the orginal image and crop the image to that size.
#Get optical flow pattern from that cropped image

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

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (9, 9),              #window size each pyramid level
	    		 maxLevel = 4,                       #number of pyramid levels 
		    	 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 10 iterations, 
                                                                                #criteria count matches the quality level
			    			4, 0.025),
                 flags = 0,                 #Flags determine what the error is calculating.
                 minEigThreshold = 1e-3)     #Threshold for smallest eignevalue minimum that can ount as good data   
                 #zero/not set flags is error from initial position at prevpts
detector = cv2.AKAZE_create()

#Create array of random colors for drawing purposes
#set the number of colors equal to the maximum number of features from feature_params
no_of_colors = 100
#create empty array with 3 rows
color_array = np.empty((0,3), int)
for n in range(no_of_colors):
  #Generate random color 
  color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
  #Add color to the array
  color_array = np.append(color_array, np.array([color]), axis=0)

def get_image(data):
  #Convert the frame's ROS image data to an opencv image
  frame = br.imgmsg_to_cv2(data)

  # create points for polygon region of interest
  points = np.array([[200, 400], [220, 380], [240, 360], [800, 360], [820, 380], [840, 400],
                      [840, 700], [200, 700]])
  # reshape array
  points = points.reshape((-1, 1, 2))

  #Create a mask
  mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
  #Draw and fill the smooth polygon
  shape = cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

  #Define the region we need
  res = cv2.bitwise_and(frame, frame, mask = mask) 
  rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rectangle around the polygon
  cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
  #Convert that frame to a grayscale image:
  gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

  #Return the cropped image in color and in grayscale
  return cropped, gray_frame

def callback(data):
  #Make the global variables for the named variables
  global prev_gray
  global key_points
  global lk_params
  global i
  global br
  global mask
  global descsi
  #Initialize cv image convertor and AKAZE feature detector
  br = CvBridge()

  #Case 1/2: it's the first frame of the whole series. 
  if i<1:
    #Take the first image
    prev_frame, prev_gray = get_image(data)

    #AKAZE detect keypoints from the first frame
    key_points, descsi = detector.detectAndCompute(prev_gray, None)
    print("What raw keypoints looks like:")
    print(key_points)
    print("array shape of raw keypoints", np.shape(key_points))
    print("what descsi looks like:")
    print(descsi)
    print("array shape of descs: ", np.shape(descsi))
    #Convert key points to coordinate tuples
    key_points = cv2.KeyPoint_convert(key_points)
    print("what converted keypoints looks like")
    print(key_points)
    print("array shape coverted keypoints: ", np.shape(key_points))

    #Find the size of the array
    array_size = int(key_points.size)
    my_size = int(array_size/2)
    #Reshape the array to be used by optical flow calculation
    key_points.shape = (my_size, 1, 2)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(prev_gray)
    i = i+1

  else:
    #Get the current frame
    cur_frame, cur_gray = get_image(data)
    cv2.imshow("current grasyscale", cur_gray)

    mask = np.zeros_like(cur_frame)

	  #calculate optical flow
    cur_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,
									        cur_gray,
									        key_points, None,
									         **lk_params)
    #Use the resizing array method again on the current points
    cur_size = int(cur_points.size)
    cur_size = int(cur_size/2)
    cur_points.shape = (cur_size, 1, 2)

    ####Make something to do with the status here
    cur_descs = detector.detectAndCompute(cur_gray, None, cur_points)

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
            mask_line = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), 
                            rand_color, 2)
            mask_circle = cv2.circle(cur_frame, (int(a), int(b)), 5,
                           rand_color, -1)
    #avg_err= sum(L1)/s
    #print("avg_err = ", avg_err)   
    image = cv2.add(mask_line, mask_circle)
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
  rospy.Subscriber('/sonar_oculus_node/M1200d/image', Image, callback)
  rospy.spin()
  cv2.destroyAllWindows()

if __name__=='__main__':
  receive_message()
