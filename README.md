# optical_flow
BlueROV Project optical flow repo

sonar_optical_flow.py 
Subscribe to a sonar image and get an image with the optical flow motion on it.
is the active version of the sonar apparent motion tracker. The code uses AKAZE feature detection to pick important features in the first frame. Then it uses the Lukas Kanade Sparse Optical Flow method to track the new locations of each key feature. Each point gets its own color and keeps that color as the flow is tracked. The motion history is not erased

oculus_viewer_2.py 
is necessary to run while using oculus bag files like sample_data.bag. I normally just leave it running while I work with any of these codes

akaze_every_frame.py 
is progress saves of the next step of improving sonar_optical_flow. That step is choosing new akaze important features any frame it's necessary so that the frames do not run out of points to keep tracking after a while
