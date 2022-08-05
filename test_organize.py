#!/usr/bin/env python3

from colorsys import rgb_to_hls
import numpy as np
import PIL

import matplotlib.pyplot as plt
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import random
from statistics import mean
import yaml

#choose a window size for the optical flow parameters (multiple of 3)
window_leg = 18

#define a global counter variable and set as 0. 
i = 0

#Create an empty list to keep the points we're tracking   
track_points_list = list()
#Create an empty dictionary that organizes all the points we're tracking
track_points_dict = {}

# Parameters for lucas kanade optical flow
lk_params = dict(winSize = (window_leg, window_leg),              #window size each pyramid level
	    		 maxLevel = 4,                       #number of pyramid levels 
		    	 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,   #termination criteria: Stop after 4 iterations, 
                                                                                #criteria count matches the quality level
			    			4, 0.025),
                 flags = 0,                 #Flags determine what the error is calculating.
                 minEigThreshold = 1e-2)     #Threshold for smallest eignevalue minimum that can ount as good data   
                 #zero/not set flags is error from initial position at prevpts

#Function and parameters for AKAZE feature detector
detector = cv2.AKAZE_create(
                  descriptor_type= 2,   #Descriptor type 2 = invariant to rotation
                                      #Descriptor type 3 = normal/not that
                  descriptor_channels=1     )

#Not using this right now after all:
#Function for Brute Force Feature Matcher
#bf = cv2.BFMatcher(cv2.NORM_L1,       #AKAZE type of descriptors
#                   crossCheck=True)  #Check that the best match from image 1 to 2 is the same as from image 2 to 1

def get_image(data):
  #Convert the frame's ROS image data to an opencv image
  frame = br.imgmsg_to_cv2(data)

  # create points for polygon region of interest
  points = np.array([[200, 180], [230, 250], [480, 180],
                      [675, 470], [200, 470]]) #Pentagon cropped to fence position
  # reshape array
  points = points.reshape((-1, 1, 2))

  # Create a mask
  mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
  # Draw and fill the smooth polygon
  shape = cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)

  # Define the region we need
  res = cv2.bitwise_and(frame, frame, mask = mask) 
  rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rectangle around the polygon
  cropped_small = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
  # Resize the cropped image so we can see it big
  # Name the rescale size
  scale_percent = 300
  # Find the rescaled width and height of the image
  width = int(cropped_small.shape[1] * scale_percent / 100)
  height = int(cropped_small.shape[0] * scale_percent / 100)

  # Resize the image
  cropped = cv2.resize(cropped_small, [width, height])
  #Convert that frame to a grayscale image:
  gray_frame = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

  #Return the cropped image in color and in grayscale
  return cropped, gray_frame

def translation(prev_x, prev_y, cur_x, cur_y):
    delta_x = cur_x - prev_x
    delta_y = cur_y - prev_y

    return delta_x, delta_y

def callback(data):
  #Make the global variables for the named variables
  global prev_gray
  global prev_frame
  global key_points
  global raw_key_points
  global lk_params
  global i
  global br
  global draw_mask
  global mask_line
  global frame
  global descsi
  global color_array
  global track_points_dict
  #Initialize cv image convertor and AKAZE feature detector
  br = CvBridge()

  #Case 1/2: it's the first frame of the whole series. 
  if i<1:
    #Take the first image
    prev_frame, prev_gray = get_image(data)
    #AKAZE detect keypoints from the first frame
    raw_key_points, descsi = detector.detectAndCompute(prev_gray, None)
    #Convert key points to coordinate tuples
    key_points = cv2.KeyPoint_convert(raw_key_points)
    #For us to see how many keypoints we have
    print("array shape coverted keypoints: ", np.shape(key_points))

    #Create array of random colors for drawing purposes
    #set the number of colors equal to the number of key_points
    no_of_colors = len(key_points)
    #create empty array with 3 rows
    color_array = np.empty((0,3), int)
    for n in range(no_of_colors):
      #Generate random color 
      color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
      #Add color to the array
      color_array = np.append(color_array, np.array([color]), axis=0)

    #Find the size of the array
    array_size = int(key_points.size)
    my_size = int(array_size/2)
    #Reshape the array to be used by optical flow calculation
    key_points.shape = (my_size, 1, 2)
    
    #add 1 to the counter so we move on from this loop
    i = i+1

  #case 2/2: it's not the first frame and we want the optical flow
  else:
    #Get the current frame
    cur_frame, cur_gray = get_image(data)
 #   cv2.imshow("current grasyscale", cur_gray)
 #   cv2.waitKey(1)
    # Create a mask image for drawing purposes
    draw_mask = np.zeros_like(cur_frame)

	  #calculate optical flow
    cur_points, st, err = cv2.calcOpticalFlowPyrLK(prev_gray,
									        cur_gray,
									        key_points, None,
									         **lk_params)

    #current points to be used as coordinates
    cur_size = int(cur_points.size)
    cur_size = int(cur_size/2)
    cur_points.shape = (cur_size, 1, 2)
    # Say a good point is one that has status 1 (status 0 means no optical flow could be found for that point
    #                                             so we don't want those ones)
    good_cur = cur_points[st == 1]
    good_prev = key_points[st == 1]

    #Make lists for the good travel distances so we can calculate the distance traveled
    good_travel_x = list()
    good_travel_y = list()

    #To check that the good points make sense:
    #Verify that the new point is reasonably close to the old point
    #For each point in good points:
    for gp in range(len(good_cur)):
        #print("cp: ", cp)
        #print(cur_points[cp][0][0], cur_points[cp][0][1])

        #print("keypoint", key_points[cp][0][0], key_points[cp][0][1])
        #Call the translation function to find the x and y translations
        travel_x, travel_y = translation(key_points[gp][0][0], key_points[gp][0][1], 
                                        cur_points[gp][0][0], cur_points[gp][0][1])

        #If it has moved 2 pixels in either x direction:
        if travel_x > 2 or travel_x < -2:
            #Too far in one frame; status is bad
            st == 0
        #If it has moved 3 pixels in eiter y direction:
        if travel_y > 3 or travel_y < -3:
            #Too far in one frame; status is bad
            st == 0
        #How abs val 2 and 3 were chosen:
        #the distance while the wall went spinning (travel x and y) were printed for a few frames
        #The highest amounts were less than 2, and y direction can have more slack than x direction 
        #since the sonar is really eucludian so 3 was chosen to make y's threshold higher
        else:
            #We want to keep the good points and the transform data
            #If the keypoint(previous point) is already anywhere in the list (as the previous frame's current point:)
            if key_points[gp][0][0] in track_points_list:
              pass
            if key_points[gp][0][1] in track_points_list:
              pass
            #If the keypoint doesn't already exist, add the tracking data to the list
            else:
              track_points_list.append(cur_points[gp][0][0])
              track_points_list.append(cur_points[gp][0][1]) 

              #add the good points' travel values to the list of good travel values
              good_travel_x.append(travel_x)
              good_travel_y.append(travel_y)

              #And add the prev points, current points, and a color to the track points dictionary
              track_points_dict[gp] = (('prev_point: ', (key_points[gp][0][0], key_points[gp][0][1])),
                                      ('cur_point: ', (cur_points[gp][0][0], cur_points[gp][0][1])),
                                      'color: ', color_array[gp].tolist())
            
   # print("the dictionary")
   # print(track_points_dict)

    print("the length of track_points dictionary is")
    print(len(track_points_dict))


 
    key_points = cur_points
    prev_gray = cur_gray
    prev_frame = cur_frame
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