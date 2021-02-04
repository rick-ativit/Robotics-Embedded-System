import argparse    # import argparse module 
import numpy as np # import Numpy module.
import cv2         # import OpenCV module.
from matplotlib import pyplot as plt # import matplotlib

# "imread" : read the image files using file name
image = cv2.imread('Figures/Birds.jpg')
cv2.imshow("Original Image", image)

# "cvtColor": convert image from one color space to another color space
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# GaussianBlur smooths image with Kernel Size = 7x7
blurred = cv2.GaussianBlur(image,(5,5),0)
cv2.imshow("Gaussian Blur", blurred)
(T,thresh)=cv2.threshold(blurred, 150,255,cv2.THRESH_BINARY_INV)

# "findContours" computes the object contour from binary image
# from external horizontal, vertical, and diagonal line segments 
cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("There are %d birds in this image" % (len(cnts)))
image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
birds = image.copy()
cv2.drawContours(birds, cnts, -1, (0,255,0), 2)
cv2.imshow("Birds", birds)
cv2.imwrite('Figures/BirdContour.jpg',birds)
# "waitKey(0) displays image for infinite time until Esc key is pressed.
cv2.waitKey(0)
# "destroyAllWindows" for closing all windows 
cv2.destroyAllWindows()
