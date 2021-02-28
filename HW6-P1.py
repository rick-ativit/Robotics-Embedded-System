import numpy as np 
import cv2         
from matplotlib import pyplot as plt 

# Image reading
frame1 = cv2.imread('Figures\SoccerTrainFrame1.png')
frame2 = cv2.imread('Figures\SoccerTrainFrame2.png')

# Step 1
diff = cv2.absdiff(frame1, frame2)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY) 

# Step 2
blurred = cv2.GaussianBlur(diff_gray, (35,35) , 0) 
_, thresh = cv2.threshold(blurred, 35, 255, cv2.THRESH_BINARY)
plt.subplot(321), plt.imshow(blurred,cmap = 'gray'), plt.title("Blur Grayscale Image")
plt.subplot(322), plt.imshow(thresh,cmap = 'gray'), plt.title("Thresholded Image")

# Step 3 (Iteration)
ErodeImg = cv2.erode(thresh, None, iterations = 5) 
DilateImg = cv2.dilate(ErodeImg, None, iterations = 10)

plt.subplot(323), plt.imshow(ErodeImg,cmap = 'gray'), plt.title("Eroded Image")
plt.subplot(324), plt.imshow(DilateImg,cmap = 'gray'), plt.title("Dilated Image")

# Step 4
contours, _ = cv2.findContours(DilateImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(frame1, contours, -1, (0,0,255), 2)
cv2.drawContours(frame2, contours, -1, (0,0,255), 2)
plt.subplot(325), plt.imshow(frame1[:,:,::-1]), plt.title("Soccer Training Frame 1")
plt.subplot(326), plt.imshow(frame2[:,:,::-1]), plt.title("Soccer Training Frame 2")
plt.show()

