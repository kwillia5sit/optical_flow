#!/usr/bin/env python3

#Import libraries and Initialize ROS Node
from cgitb import grey
from dataclasses import dataclass
from sys import _current_frames
import rospy
from sensor_msgs.msg import Image  
#Image is the message/topic type
import numpy as np
from cv_bridge import CvBridge
#CvBridge converts between ROS and Opencv images
import cv2


i = 0 #initializes a counter variable
print("Ready")

def callback(data):
  #callback(data) is like the loop of code that the main 
  #code is calling into use
  br = CvBridge() #converts from ros image to opencv image
 
  #Tell silly python these three are global variables
  global i
  global prev_gray
  global firstframe

  #condition 1/2:
  #for the first frame, there's nothing to compare
  #so just save that one to be the next pre_gray
  if i < 1:
    firstframe = br.imgmsg_to_cv2(data)
    #converts first frame to opencv image
    prev_gray = cv2.cvtColor(firstframe, cv2.COLOR_BGR2GRAY)
    #converts color image to grayscale image, call it prev_gray
    rospy.loginfo("viewed first image")
    #debug to the terminal
    i = i+1
    #add 1 to the i counter
    cv2.waitKey(2)
    #move to next image after 1 millisecond

  #condition 2/2 (all else):
  #each gray should be compared to the prev_gray
  #in the function for dense optflow
  else:    
    #if you need to debug uncomment: 
    cv2.imshow("first prev_gray", prev_gray)
    current_frame = br.imgmsg_to_cv2(data)
    #converts ROS Image message to OpenCV image
    rospy.loginfo("recieving video frame, ")
    #output debugging info to the terminal
    #Display current image:
    cv2.imshow("camera", current_frame)

    # Creates an image filled with zero
    # intensities with the same dimensions 
    # as the frame
    mask = np.zeros_like(current_frame)
    # Sets image saturation to maximum
    mask[..., 1] = 255

    # Converts current image to grayscale
    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)


    #Read frame compared to last frame
    #DOING THE OPTICAL FLOW
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 
                                        #prevImg, current Img,
                                        None, 
                                        #flow: computed flow image that has same size as previous frame,
                                        0.25, 
                                        #pyr_scale: how much smaller the pyramid layer is from
                                        # the previous one
                                        # Keep it small because at 0.5, sometimes the next layers weren't 
                                        # small enough to see the difference between frames, causing black screens.
                                        3, 
                                        #levels: number of pyramid layers, inclduing first image
                                        18, 
                                        #winsize: avg window size. Larger numbers detct more noise and
                                        # detect faster movements, but make movement area look blurrier
                                        3, 
                                        #iterations: number of iterations at each pyramid layer
                                        7, 
                                        #polyN: size of pixel neighborhood used for polynomial expansions
                                        # Larger numbers are more robust algorithms, can approximate smoother 
                                        # surfaces, and make blurrier outputs
                                        1.5, 
                                        #Standard deviation of the Gaussian for the polynomial expansions
                                        1)
                                        #Operation flags
      # Compute the magnitude and angle of the 2D vectors                                
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Sets image hue according to the optical flow 
    # direction
    mask[..., 0] = angle * 180 / np.pi / 2

    # Sets image value according to the optical flow
    # magnitude (normalized)
    mask[..., 2] = cv2.normalize(magnitude, magnitude, 0, 255, cv2.NORM_MINMAX)

    # Converts HSV to RGB (BGR) color representation
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

    # Opens a new window and displays the output frame
    cv2.imshow("dense optical flow", rgb)

    #turn the gray image just used into the next prev_gray
    prev_gray = gray
    #if you need to debug uncomment: 
    cv2.imshow("new prev_gray", prev_gray)
    #add 1 to the counter
    i = i+1

    #millisecond delay before next image rolls in
    cv2.waitKey(2) 
  

# runs once
def receive_message():
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
    rospy.init_node('optflow_node', anonymous=True)
  #Convert first frame from ros data to opencv data (image)
  # Node is subscribing to the video_frames topic
    rospy.Subscriber('/sonar_oculus_node/M1200d/image', Image, callback)  

    rospy.spin()

    cv2.destroyAllWindows()


if __name__=='__main__':
  receive_message()
#main code, says to run receive_message()