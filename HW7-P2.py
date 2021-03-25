import numpy as np # import NumPy module
import cv2         # import OpenCV module.
from matplotlib import pyplot as plt

# Initiate video capture for video file
cap = cv2.VideoCapture('Figures/PizzaConveyor.mp4')
frame = None      # each frame from VDO file
roiPts = []       # 4 corner point of ROI
inputMode = False # Check for mouse-click input mode
saved = []

# Callback function to create ROI from 4 points, clicked by the user 
def click_and_crop(event, x, y, flags, param):
    # Declare global variables: 1)current frame, 2) list of ROI points, 3) input mode
    global frame, roiPts, inputMode, saved
 
    # Checking conditions: ROI selection/input mode, left-mouse click, if 4 points are selected or not?
    if inputMode and event == cv2.EVENT_LBUTTONDOWN and len(roiPts) < 4:
        roiPts.append((x, y)) # update a list of ROI points with (x, y) location of mouse click
        cv2.circle(frame, (x, y), 4, (0, 255, 0), 2) # draw green circle at mouse click (x,y) location
        cv2.imshow("image", frame)

# Attaching the callback into the video window
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
roiBox = None # intialize ROI region as none/empty

while (True): # Main loop
    ret, frame = cap.read() # Reading next frames from video
    frame1 = frame.copy()
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    fps=cap.get(cv2.CAP_PROP_POS_FRAMES)
    if roiBox is None and len(roiPts) < 4:
        inputMode = True # indicate that we are in input mode 
        orig = frame.copy() # copy the first frame of VDO file 
        # keep looping until 4 reference-corner points are selected for ROI;
        # press any key to exit ROI selction mode once 4 points have been selected
        while len(roiPts) < 4:
            cv2.imshow("image", frame)
            cv2.waitKey(0)
        # determine the top-left and bottom-right points
        roiPts = np.array(roiPts)
        s = roiPts.sum(axis = 1)
        tl = roiPts[np.argmin(s)] # tl denotes top-left corner point
        br = roiPts[np.argmax(s)] # br denotes bottom-right corner point
         
        roi = orig[tl[1]:br[1], tl[0]:br[0]] # crop ROI-bounding box from original frame
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # convert ROI box to HSV-color space
        # compute 2D histogram for ROI box from Hue-&Value-channels
        roi_hist = cv2.calcHist([roi],[0,2], None, [180,256], [0,180,0,255])
        # setup an initial search window for meanshift method
        roiBox = (tl[0], tl[1], br[0]-tl[0], br[1]-tl[1])
        x,y,w,h = roiBox
        xc = int(x+(w/2))
        yc = int(y+(h/2))
        saved = np.array([(xc,yc,w,h,fps)])
        saved1 = np.copy(saved)
        roiBox1 = roiBox
    elif roiBox is not None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert current frame to HSV-color space
        
        backProj = cv2.calcBackProject([hsv],[0,2],roi_hist,[0,180,0,255],scale=2) # compute backproject from roi_hist
        
        ret, roiBox = cv2.meanShift(backProj, roiBox, term_crit) # "meanshift": compute the new window to track object
        ret, roiBox1 = cv2.CamShift(backProj, roiBox1, term_crit) # "meanshift": compute the new window to track object

        x,y,w,h = roiBox
        xc = int(x+(w/2))
        yc = int(y+(h/2))
        saved = np.append(saved, [(xc,yc,w,h,fps)], axis=0)
        
        x1,y1,w1,h1 = roiBox1
        xc1 = int(x1+(w1/2))
        yc1 = int(y1+(h1/2))
        saved1 = np.append(saved1, [(xc1,yc1,w1,h1,fps)], axis=0)
#         print("x roi = ",x," y roi = ",y)
        frame = cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0) ,2) # Draw new windown on current frame
        cv2.circle(frame, (int(x+(w/2.0)), int(y+(h/2.0))), 4, (0, 255, 0), 2) # Draw a circle center of new window 
        frame1 = cv2.rectangle(frame1, (x1,y1), (x1+w1,y1+h1), (0,0,255) ,2) # Draw new windown on current frame
        cv2.circle(frame1, (int(x1+(w1/2.0)), int(y1+(h1/2.0))), 4, (0, 0, 255), 2) # Draw a circle center of new window 
        
        
    cv2.imshow("Object Tracking: MeanShift", frame)
    cv2.imshow("Object Tracking: CamShift", frame1)
    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27 or fps==338:
        break        
cv2.destroyAllWindows()
cap.release()
fps = saved[...,4]
xc = saved[...,0]
yc = saved[...,1]
w = saved[...,2]
h = saved[...,3]
xc1 = saved1[...,0]
yc1 = saved1[...,1]
w1 = saved1[...,2]
h1 = saved1[...,3]
plt.subplot(221)
plt.title("X Center: Mean Shift")  
plt.plot(fps,xc)
plt.subplot(222)
plt.title("Y Center: Mean Shift")  
plt.plot(fps,yc)
plt.subplot(223)
plt.title("ROI Width: Mean Shift")  
plt.plot(fps,w)
plt.subplot(224)
plt.title("ROI Height: Mean Shift")  
plt.plot(fps,h)
plt.show()

plt.subplot(221)
plt.title("X Center: Cam Shift")  
plt.plot(fps,xc1)
plt.subplot(222)
plt.title("Y Center: Cam Shift")  
plt.plot(fps,yc1)
plt.subplot(223)
plt.title("ROI Width: Cam Shift")  
plt.plot(fps,w1)
plt.subplot(224)
plt.title("ROI Height: Cam Shift")  
plt.plot(fps,h1)
plt.show()

fig = plt.figure()
plt.subplot(221)
plt.title("X Center: Compare")  
l1,=plt.plot(fps,xc)
l2,=plt.plot(fps,xc1)
plt.subplot(222)
plt.title("Y Center: Compare")  
plt.plot(fps,yc,fps,yc1)
plt.subplot(223)
plt.title("ROI Width: Compare")  
plt.plot(fps,w,fps,w1)
plt.subplot(224)
plt.title("ROI Height: Compare")  
plt.plot(fps,h,fps,h1)
fig.legend((l1,l2),('Mean Shift','Cam Shift'),loc = "upper center")
plt.show()