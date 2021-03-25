import cv2         # import OpenCV module.
from matplotlib import pyplot as plt # import matplotlib

# Step 1
img = cv2.imread('Figures/PizzaOnConveyor.jpg')
img_ref = img.copy()
# "selectROI" : used to select a single RegionOfInterest bounding box.
roiTop, roiLeft, roiWidth, roiHeight = cv2.selectROI(img)
print(roiTop, roiLeft, roiWidth, roiHeight)
roi = img[roiLeft:roiLeft+roiHeight, roiTop:roiTop+roiWidth]
cv2.rectangle(img_ref, (roiTop,roiLeft), (roiTop+roiWidth,roiLeft+roiHeight), (0,0,255), 2)

# Step 2
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# 1D histogram for img HSV
histH_img = cv2.calcHist([img_hsv], [0], None, [180], [0,180])
histS_img = cv2.calcHist([img_hsv], [1], None, [256], [0,255])
histV_img = cv2.calcHist([img_hsv], [2], None, [256], [0,255])
# 2D histogram for img HSV
hist2D_img = cv2.calcHist([img_hsv], [0,2], None, [180,256], [0,180,0,255])
cv2.normalize(hist2D_img, hist2D_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);

roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 1D histogram for ROI HSV
histH_roi = cv2.calcHist([roi_hsv], [0], None, [180], [0,180])
histS_roi = cv2.calcHist([roi_hsv], [1], None, [256], [0,255])
histV_roi = cv2.calcHist([roi_hsv], [2], None, [256], [0,255])
# 2D histogram for ROI HSV
hist2D_roi = cv2.calcHist([roi_hsv], [0,2], None, [180,256], [0,180,0,255])
cv2.normalize(hist2D_roi, hist2D_roi, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX);
# Plot and visualized Histogram for selecting 2 channels
plt.subplot(241)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.subplot(242)
plt.plot(histH_img)
plt.title("img: Hue Histogram")
plt.subplot(243)
plt.plot(histS_img)
plt.title("img: Sat Histogram")
plt.subplot(244)
plt.plot(histV_img)
plt.title("img: Val Histogram")
plt.subplot(245)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.subplot(246)
plt.plot(histH_roi)
plt.title("ROI: Hue Histogram")
plt.subplot(247)
plt.plot(histS_roi)
plt.title("ROI: Sat Histogram")
plt.subplot(248)
plt.plot(histV_roi)
plt.title("ROI: Val Histogram")
plt.show()
# Plot 2D histogram from chosen channels (H&V)
plt.subplot(221)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.subplot(222)
plt.imshow(cv2.cvtColor(hist2D_img, cv2.COLOR_BGR2RGB))
plt.title("Original Image: 2D Histogram")
plt.xlabel('Value')
plt.ylabel('Hue')
plt.subplot(223)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.subplot(224)
plt.imshow(cv2.cvtColor(hist2D_roi, cv2.COLOR_BGR2RGB))
plt.title("ROI: 2D Histogram")
plt.xlabel('Value')
plt.ylabel('Hue')
plt.show()

# Step 3
# Compute BackProject image
backproj = cv2.calcBackProject([img_hsv], [0,2], hist2D_roi, [0,180,0,255], scale=2)
# Plot back projection
plt.subplot(121)
plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
plt.title("Original Image w/ ROI")
plt.subplot(122)
plt.imshow(cv2.cvtColor(backproj, cv2.COLOR_BGR2RGB))
plt.title("Back Projection")
plt.show()
