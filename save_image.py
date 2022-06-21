#Will use image_sub and save to rotate.mp4
# #!/usr/bin/env python3

#Import libraries and Initialize ROS Node
from sys import _current_frames
import rospy
from sensor_msgs.msg import Image  
#Image is the message/topic type
import numpy as np
from cv_bridge import CvBridge
#CvBridge converts between ROS and Opencv images
import cv2
cap = cv2.VideoCapture(0)

def callback(data):
    br = CvBridge() #converts 

    rospy.loginfo("recieving video frame, ")
    #output debugging info to the terminal

    current_frame=br.imgmsg_to_cv2(data)
    #converts ROS Image message to OpenCV image
    
    #Display current image:
    cv2.imshow("camera", current_frame)
    
    #save variables
    codec = cv2.VideoWriter_fourcc(*'XVID')
    #make frame measurements
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(5))
    resolution = (width, height)
    #(video file name, fourcc codec, frame rate, resolution)
    save = cv2.VideoWriter('Rotate.avi', codec, fps, resolution)
    save.write(current_frame)
    rospy.loginfo(' frame saved')


    #millisecond delay before next frame comes in
    cv2.waitKey(1)


def receive_message():
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('optflow_node', anonymous=True)

 # Node is subscribing to the video_frames topic
  rospy.Subscriber('/sonar_oculus_node/M750d/image', Image, callback)  
  
#keep going until frames run out
  rospy.spin()

  #close down video stream when done
  cv2.destroyAllWindows()
  cap.release()
  
 
if __name__=='__main__':
 receive_message()


