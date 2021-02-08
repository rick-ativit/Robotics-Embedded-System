import argparse    # import argparse module 
import numpy as np # import Numpy module.
import cv2         # import OpenCV module.
from matplotlib import pyplot as plt # import matplotlib

# Step 1
image = cv2.imread('Figures/BlueScreenJump.jpg')
cv2.imshow("Original Image", image)
imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
cv2.imshow("LAB Image", imageLAB)
blurred = cv2.GaussianBlur(imageLAB,(7,7),0)
cv2.imshow("Gaussian Blur", blurred)
# Step 2
(L, A, B) = cv2.split(blurred)
histL = cv2.calcHist([L], [0], None, [256], [0,256])
histA = cv2.calcHist([A], [0], None, [256], [0,256])
histB = cv2.calcHist([B], [0], None, [256], [0,256])
plt.figure()
plt.title("LAB Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(histL,"b")
plt.plot(histA,"r")
plt.plot(histB,"g")
plt.legend(('L','A','B'))
plt.xlim([0,256])
plt.show()
# Step 3
(T,thresh)=cv2.threshold(B, 100,255,cv2.THRESH_BINARY)

#Step 4
cnts, hierarchy1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

imageC = image.copy()
cv2.drawContours(imageC, cnts, -1, (0,0,255), 2)
cv2.imshow("Contour Image", imageC)
cv2.imwrite('Figures/Contour.jpg',imageC)
