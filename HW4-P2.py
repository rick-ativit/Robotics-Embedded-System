import cv2          
import numpy as np  
from matplotlib import pyplot as plt 

# Import image (Gray scale)
image = cv2.imread('Figures\CarOriginal.jpg',0)
hSI, wSI = image.shape[:2] 
print("Original Image: width=",wSI,",height=",hSI)
template = cv2.imread('Figures\CarTemplate.jpg',0)
hT, wT = template.shape[:2] 
print("Template Image: width=",wT,",height=",hT)

# 6 methods
methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCOEFF', 
           'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED']

# Matchin loop
for meth in methods:
    image_loop = image.copy()
    method = eval(meth) 
    print("Method = ",methods[eval(meth)])
    Mresult = cv2.matchTemplate(image_loop,template,method)
    hM, wM = Mresult.shape[:2]
    print("Template Matching Image: width=",wM,",height=",hM)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(Mresult)
    print("minV = ",min_val,", maxV = ",max_val)

    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else: 
        top_left = max_loc
    bottom_right = (top_left[0] + wT, top_left[1] + hT)
    cv2.rectangle(image_loop,top_left, bottom_right, (255,255,255), 2)

    plt.subplot(221),plt.imshow(Mresult,cmap = 'gray')
    plt.title('Template Matching Result'), plt.xticks([]), plt.yticks([])
    plt.subplot(222),plt.imshow(image_loop,cmap = 'gray')
    plt.title('Best Match Area'), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
    
