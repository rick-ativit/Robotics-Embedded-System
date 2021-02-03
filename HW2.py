import numpy as np                   # import Numpy module.
import cv2                           # import OpenCV module.
from matplotlib import pyplot as plt # import matplotlib
# Read the image files for Pink,Green,Blue
imageP = cv2.imread('Figures/PinkPaperColor.jpg')
imageG = cv2.imread('Figures/GreenPaperColor.jpg')
imageB = cv2.imread('Figures/BluePaperColor.jpg')
# Convert the image files to HSV
imageP_HSV = cv2.cvtColor(imageP, cv2.COLOR_BGR2HSV)
imageG_HSV = cv2.cvtColor(imageG, cv2.COLOR_BGR2HSV)
imageB_HSV = cv2.cvtColor(imageB, cv2.COLOR_BGR2HSV)

# Mask for each image
maskP = np.zeros(imageP.shape[:2], np.uint8) 
maskP[80:280, 150:350] = 255
maskG = np.zeros(imageG.shape[:2], np.uint8) 
maskG[70:285, 170:400] = 255
maskB = np.zeros(imageB.shape[:2], np.uint8) 
maskB[80:240, 115:275] = 255
# Masked image
masked_imageP = cv2.bitwise_and(imageP,imageP,mask = maskP)
masked_imageG = cv2.bitwise_and(imageG,imageG,mask = maskG)
masked_imageB = cv2.bitwise_and(imageB,imageB,mask = maskB)
# Masked image (HSV)
masked_imageP_HSV = cv2.cvtColor(masked_imageP, cv2.COLOR_BGR2HSV)
masked_imageG_HSV = cv2.cvtColor(masked_imageG, cv2.COLOR_BGR2HSV)
masked_imageB_HSV = cv2.cvtColor(masked_imageB, cv2.COLOR_BGR2HSV)

# Split Channel for images
(Hue_P, Sat_P, Val_P) = cv2.split(imageP_HSV)
(Hue_G, Sat_G, Val_G) = cv2.split(imageG_HSV)
(Hue_B, Sat_B, Val_B) = cv2.split(imageB_HSV)
(Hue_PM, Sat_PM, Val_PM) = cv2.split(masked_imageP_HSV)
(Hue_GM, Sat_GM, Val_GM) = cv2.split(masked_imageG_HSV)
(Hue_BM, Sat_BM, Val_BM) = cv2.split(masked_imageB_HSV)

# CalcHist
histHue_P = cv2.calcHist([Hue_P], [0], None, [256], [0,256])
histSat_P = cv2.calcHist([Sat_P], [0], None, [256], [0,256])
histVal_P = cv2.calcHist([Val_P], [0], None, [256], [0,256])

histHue_PM = cv2.calcHist([Hue_PM], [0], maskP, [256], [0,256])
histSat_PM = cv2.calcHist([Sat_PM], [0], maskP, [256], [0,256])
histVal_PM = cv2.calcHist([Val_PM], [0], maskP, [256], [0,256])

histHue_G = cv2.calcHist([Hue_G], [0], None, [256], [0,256])
histSat_G = cv2.calcHist([Sat_G], [0], None, [256], [0,256])
histVal_G = cv2.calcHist([Val_G], [0], None, [256], [0,256])

histHue_GM = cv2.calcHist([Hue_GM], [0], maskG, [256], [0,256])
histSat_GM = cv2.calcHist([Sat_GM], [0], maskG, [256], [0,256])
histVal_GM = cv2.calcHist([Val_GM], [0], maskG, [256], [0,256])

histHue_B = cv2.calcHist([Hue_B], [0], None, [256], [0,256])
histSat_B = cv2.calcHist([Sat_B], [0], None, [256], [0,256])
histVal_B = cv2.calcHist([Val_B], [0], None, [256], [0,256])

histHue_BM = cv2.calcHist([Hue_BM], [0], maskB, [256], [0,256])
histSat_BM = cv2.calcHist([Sat_BM], [0], maskB, [256], [0,256])
histVal_BM = cv2.calcHist([Val_BM], [0], maskB, [256], [0,256])

#Plot
plt.subplot(341), plt.imshow(imageP)
plt.subplot(342), plt.imshow(masked_imageP)
plt.subplot(343), plt.plot(histHue_P,"b"),plt.plot(histSat_P,"r"),plt.plot(histVal_P,"g")
plt.subplot(344), plt.plot(histHue_PM,"b"),plt.plot(histSat_PM,"r"),plt.plot(histVal_PM,"g")
plt.subplot(345), plt.imshow(imageG)
plt.subplot(346), plt.imshow(masked_imageG)
plt.subplot(347), plt.plot(histHue_G,"b"),plt.plot(histSat_G,"r"),plt.plot(histVal_G,"g")
plt.subplot(348), plt.plot(histHue_GM,"b"),plt.plot(histSat_GM,"r"),plt.plot(histVal_GM,"g")
plt.subplot(349), plt.imshow(imageB)
plt.subplot(3,4,10), plt.imshow(masked_imageB)
plt.subplot(3,4,11), plt.plot(histHue_B,"b"),plt.plot(histSat_B,"r"),plt.plot(histVal_B,"g")
plt.subplot(3,4,12), plt.plot(histHue_BM,"b"),plt.plot(histSat_BM,"r"),plt.plot(histVal_BM,"g")

plt.xlim([0,256])
plt.show()

for i in range(0,255):
    print(i,"\t","Pink Hue",'\t',histHue_P[i])
    print(i,"\t","Pink Sat",'\t',histSat_P[i])
    print(i,"\t","Pink Value",'\t',histVal_P[i])    
    print(i,"\t","Green Hue",'\t',histHue_G[i])
    print(i,"\t","Green Sat",'\t',histSat_G[i])
    print(i,"\t","Green Value",'\t',histVal_G[i])
    print(i,"\t","Blue Hue",'\t',histHue_B[i])
    print(i,"\t","Blue Sat",'\t',histSat_B[i])
    print(i,"\t","Blue Value",'\t',histVal_B[i])