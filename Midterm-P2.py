import numpy as np 
import cv2        
from matplotlib import pyplot as plt 
import imreg_dft as ird

# Step 1: Load and convert Image
Template = cv2.imread('TemplatePart.jpg')
Defected = cv2.imread('DefectedPartConveyor.jpg')
Ref = Defected.copy() #Copy as reference for another step
Defected_LAB = cv2.cvtColor(Defected, cv2.COLOR_BGR2LAB)
Template_gray = cv2.cvtColor(Template, cv2.COLOR_BGR2GRAY)
Defected_gray = cv2.cvtColor(Defected, cv2.COLOR_BGR2GRAY)
plt.subplot(221)
plt.imshow(Template_gray,cmap = 'gray')
plt.title("Template Image")
plt.subplot(222)
plt.imshow(Defected_gray,cmap = 'gray')
plt.title("Original Image")
# Step 2: Threshold in LAB space and draw contour
(L, A, B) = cv2.split(Defected_LAB)
(T,thresh)=cv2.threshold(L, 100,255,cv2.THRESH_BINARY) #Use L channel
plt.subplot(223)
plt.imshow(thresh,cmap = 'gray')
plt.title("Threshold Image")
cnts, hierarchy1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(Defected, cnts, -1, (0,0,255), 2) #Draw contour on Defected
plt.subplot(224)
plt.imshow(Defected[:,:,::-1])
plt.title("Contour Image")
# Step 3: Draw bounding boxes and segment them
boundRect = [None]*len(cnts) #Creat empty list
Bound = Defected.copy() #Copy Defected array
for i in range(len(cnts)): #Create loop for boundingRect array
    boundRect[i] = cv2.boundingRect(cnts[i])
Segment=np.copy(boundRect) #Copy boundRect list into Segment array
ROI_number = 1 #Place holder number
for i in range(len(cnts)): #Create loop for drawing rectangle
    if boundRect[i][3]>=100 and boundRect[i][2]>=100:
        Segment[i][0]=int(boundRect[i][0]-25)
        Segment[i][1]=int(boundRect[i][1]-25) 
        cv2.rectangle(Bound, (int(Segment[i][0]), int(Segment[i][1])), 
          (int(Segment[i][0]+280), int(Segment[i][1]+280)), (0,255,0), 2) #Draw bounding box
        ROI = Ref[Segment[i][1]:Segment[i][1]+280, Segment[i][0]:Segment[i][0]+280] #create ROI according to bounding box
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI) #write ROI into png image
        ROI_number += 1
cv2.imshow('Bounding-Box Image',Bound)

# Step 4: Template matching w/ similarity function
Stamp1 = cv2.imread('ROI_1.png')
Stamp2 = cv2.imread('ROI_2.png')
Stamp3 = cv2.imread('ROI_3.png')
Stamp4 = cv2.imread('ROI_4.png')
Stamp1_gray = cv2.cvtColor(Stamp1, cv2.COLOR_BGR2GRAY)
Stamp2_gray = cv2.cvtColor(Stamp2, cv2.COLOR_BGR2GRAY)
Stamp3_gray = cv2.cvtColor(Stamp3, cv2.COLOR_BGR2GRAY)
Stamp4_gray = cv2.cvtColor(Stamp4, cv2.COLOR_BGR2GRAY)

result1 = ird.similarity(Template_gray, Stamp1_gray, numiter=3)
result2 = ird.similarity(Template_gray, Stamp2_gray, numiter=3)
result3 = ird.similarity(Template_gray, Stamp3_gray, numiter=3)
result4 = ird.similarity(Template_gray, Stamp4_gray, numiter=3)

print("Result for Stamp1:")
print("Translation is {}, Rotated Angle is {:.4g}".format(result1['tvec'], result1['angle']))
print("Result for Stamp2:")
print("Translation is {}, Rotated Angle is {:.4g}".format(result2['tvec'], result2['angle']))
print("Result for Stamp3:")
print("Translation is {}, Rotated Angle is {:.4g}".format(result3['tvec'], result3['angle']))
print("Result for Stamp4:")
print("Translation is {}, Rotated Angle is {:.4g}".format(result4['tvec'], result4['angle']))

plt.show()