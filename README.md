# Automatic-6-DOF-robotic-arm-control-via-computer-vision


# Objective:
The aim of this project was two fold:
1. Apply the lessons we have learned about 6-DOF robotic controls, especially those used in the current industry, to reliably 3D print objects
2. Create a 3D visual map to get accurate and live distance measurments
<br><br>
Combined, we get a system that can not only print out various items of our choosing, but also **accurately measure the distance between a user's hand and the 3d printing component**.
This last part was crucial to our project, as we wanted to make a safe system that stopped printing if a user's hand was detected near the 3D printing pen.
<br/><br/>
This was a team project, and my contribution/role was in the **computer vision control**, combining the depth function of the Intel realsense Camera with the RGB vision for accurate measurement between hand and 3D pen location.
<br/><br/>
# Equipment used:
- Doosan Robotics 6-DOF robotic arm (E0509)
- Intel Realsense Camera (D455F)
- Sanago 3D pen
<br/><br/>
# libraries + systems used (All the code was done via Python):
- DART (for robotic arm control)
- ROS2 (for robotic arm control + server-client connection)
- YOLO segmentation (computer vision)
- Mediapipe (getting accurate location + reading of user's hand locations)
- Roboflow (training the YOLO segmentation model for additional items it needs to reliably detect,specifically the 3D pen)
<br/><br/>
<img width="766" height="401" alt="{BFFA47BC-EF38-4F0D-B62D-49572899F3BC}" src="https://github.com/user-attachments/assets/1f985693-e032-41aa-9c44-c43909191e4f" />
<br/><br/>

# How the system works:
1. Connection established between server (robot arm) and client (user computer)
2. User gives command of which shape to draw from a list of available
3. once selected a new window pops up that shows the currently location of the 3D pen in 3D coordinates (i.e. xyz coordinates)
4. if a hand is detected (via mediapipe) then the finger locations are all marked in 3D coordinates, with the closest finger to the 3D pen being shown on screen
5. The distance between the 3D pen and the closest finger is marked and measured
6. if the hand enters within **25cm** of the pen, then the system gives out warning messages to the terminal and **slow down printing speed down to 70%**
7. if the hand enters within **20cm** of the pen, then the system stops **entirely**, the robotic arm stops moving and the 3D pen stops printing
8. if the hand is then moved away from the 3D pen, then printing automatically continues
<br/><br/>
# Some imortant notes about Computer Vision (my contribution)
- Calibration between the RGB camera and the Depth camera (both part of Intel Realsense camera) was required to get accurate XYZ reading of items in view.
- Firstly, the RGB camera detects what items are in view and if it detects either a 'hand' class (Mediapipe) or a 3D pen (trained via Roboflow) then the location of the items are noted in 2D (i.e. the u,v pixel location of the hand in a 2D camera)
- then, once the u,v coordinate is noted, the system send the pixel coordinates to the depth camera, where **the depth (Z coordinate) of said pixel is detected and noted**
- now that the Z coordinate of an object, be it hand or pen, is know, the x,y coordinated had to be calculated too, this was done by the following formula:<br/>
<img width="375" height="103" alt="{65038955-0C7E-4F68-84BF-0422D2F54AD9}" src="https://github.com/user-attachments/assets/12255b68-7691-41c3-bbd0-ee6b7249e3df" /><br/><br/>
where: <br/>
X = x coordinate of object <br/>
Y = y coordiante of object <br/>
Z = Z coordinate (depth/distance from camera) of object <br/> 
u = horizontal pixel coordinate <br/>
v = vertical pixel coordinate <br/>
cx = principal point x (pixel x-coordinate of the optical center)<br/>
cy = principal point y (pixel y-coordinate of the optical center)<br/>
fy, fx =focal lengths (in pixels) <br/>
- Once the XYZ coordinate of BOTH the hand and pen was detected, distance between them was calculated via **3D Euclidean distance**, taking both XYZ coordinates of both the pen and the hand to calculate the distance between. this was one of the most important parts of our project
<img width="350" height="100" alt="{FFE16CB9-0AD0-452F-9727-CB3F6F481682}" src="https://github.com/user-attachments/assets/ced6c3d7-745c-4d09-9088-dcbcaecdbdee" />
<img width="800" height="400" alt="image" src="https://github.com/user-attachments/assets/e84ed44f-38a3-420d-b3db-7c2a35d1f375" />

- note: I did NOT do the aruco boundary part, that was added on by another member
- in case the Depth camera was down, we made a failsafe system where the distance woulde be measured via 2D camera only, ie 2D Euclidean distance calculation, while not as accurate as 3D, and false readings were common, it was safer than nothing.

