python
#save video
def write_video():
  #variables
  filename = "Rotate.mp4"
  codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  framerate = 10
  width = int(cap.get(3))
  height = int(cap.get(4))
  resolution = (width, height)
  #function
  write = cv2.VideoWriter(filename, codec, framerate, resolution)
  if cap.isOpened():
    ret, freame = cap.read()
  else:
    ret = False
  while ret:
    ret, frame = cap.read()
    write.write(frame)
    cv2.imshow("camera", frame)
    if cv2.waitKey(1) == 27:
          break
  rospy.loginfo(' frame saved')  
  rospy.spin()
