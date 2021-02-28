import numpy as np 
import cv2         

# Reading 2 consecutive frames
cap = cv2.VideoCapture('Figures\SoccerTraining.mp4') 
ret, frame1 = cap.read()
ret, frame2 = cap.read()

# Step 1
while cap.isOpened():
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 
# Step 2
    blur = cv2.GaussianBlur(gray, (35,35) , 0) 
    _, thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
# Step 3 (Iteration)
    ErodeImg = cv2.erode(thresh, None, iterations = 5) 
    DilateImg = cv2.dilate(ErodeImg, None, iterations = 10)
# Step 4    
    contours, _ = cv2.findContours(DilateImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame1, contours, -1, (0,0,255), 2)  # -1 option for draw all contours
    cv2.imshow('Moving Contour',frame1)
    frame1 = frame2 # reset frame1 using frame2
    ret, frame2 = cap.read() # read next frame (frame2) from video
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()

