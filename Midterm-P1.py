import cv2      # import OpenCV module.
import numpy as np # import NumPy module
from matplotlib import pyplot as plt 

image = cv2.imread('CountingCircle.jpg')
image2 = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("Original Image", image2)

(H, S, V) = cv2.split(image2)
histH = cv2.calcHist([H], [0], None, [256], [0,256])
histS = cv2.calcHist([S], [0], None, [256], [0,256])
histV = cv2.calcHist([V], [0], None, [256], [0,256])

plt.figure()
plt.title("H,S,V")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.plot(histH,"b"),plt.plot(histS,"r"),plt.plot(histV,"g")
plt.legend(("H","S","V"))
plt.xlim([0,256])

(T, thresh) = cv2.threshold(V, 220, 255, cv2.THRESH_BINARY)
cv2.imshow("Thresh Image", thresh)

# Creating square kernel of size (5x5) pixel  
kernel = np.ones((3, 3), np.uint8)   
# "erode" method: thining foreground objects
ErodeImg = cv2.erode(thresh, kernel)  
# "dilate" method: enlarging foreground objects
#DilateImg = cv2.dilate(ErodeImg, kernel)
cv2.imshow("Morph Image", ErodeImg)

cnts, hierarchy = cv2.findContours(ErodeImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("There are %d circles in this image" % (len(cnts)))
image3 = image.copy()
cv2.drawContours(image3, cnts, -1, (0,0,225), 1)
cv2.imshow("Contour Image", image3)
plt.show()




