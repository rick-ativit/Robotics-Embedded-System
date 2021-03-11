import numpy as np # import NumPy module
import cv2         # import OpenCV module.

# Initiate video capture for video file
cap = cv2.VideoCapture('GoalKeeperTraining.mp4')
# Reading two consecutive frames from video 
ret, frame1 = cap.read()
ret, frame2 = cap.read()
#create variable with list of value to use in absdiff and bgst operation
kernel = np.ones((3, 3), np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)) #Use elliptical kernel
backSub = cv2.createBackgroundSubtractorKNN(history = 70, dist2Threshold = 500, detectShadows = True)

while cap.isOpened():
    # "absdiff": computing a per-element absolute difference of 2 frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) # convert color- to gray-image
    blur = cv2.GaussianBlur(gray, (35,35) , 0) # smooths image with "GaussianBlur": Kernel Size = 35x35
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    # "erode" method: thining foreground objects
    ErodeImg = cv2.erode(thresh, kernel, iterations = 5)
    # "dilate" method: enlarging foreground objects
    DilateImg = cv2.dilate(ErodeImg, kernel, iterations = 25)
    # "findContours" computes the object contour from binary image
    # from external horizontal, vertical, and diagonal line segments 
    contours, _ = cv2.findContours(DilateImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1, contours, -1, (0,255,0), 2)  # -1 option for draw all contours

    # plot the moving contour
    cv2.imshow('Moving Contour',frame1)
    frame1 = frame2 # reset frame1 using frame2
    ret, frame2 = cap.read() # read next frame (frame2) from video
    
    #bgsubtrack
    ret, frame = cap.read() # Reading next frames from video 

    #update the background model
    fgMask = backSub.apply(frame)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel2)

    # display_frame_number: get the frame number and write it on the current frame
    cv2.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    fps=cap.get(cv2.CAP_PROP_POS_FRAMES)
    #show the current frame and the fg masks
    if cv2.waitKey(300) == 27 or fps==70:
        break
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Background Subtractor KNN', fgMask)

cap.release()
cv2.destroyAllWindows()