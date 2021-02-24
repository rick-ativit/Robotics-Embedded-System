import numpy as np 
import cv2        
from matplotlib import pyplot as plt 
import imreg_dft as ird 

# Step1
image = cv2.imread('Figures\StampingPart.jpg')
cv2.imshow("Original Image",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
rows,cols = image.shape[:2]
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step2
Mt = np.float32([[1,0,50],[0,1,50]])
template = cv2.warpAffine(image_gray,Mt,(cols+800,rows+400))
rows_t,cols_t = template.shape[:2]

# Step3
Mr = cv2.getRotationMatrix2D((cols/2,rows/2),25,1)
rot = cv2.warpAffine(image_gray,Mr,(cols,rows))
rows_r,cols_r = rot.shape[:2]
Mrt = np.float32([[1,0,100],[0,1,400]])
source = cv2.warpAffine(rot,Mrt,(cols+800,rows+400))
rows_rt,cols_rt = source.shape[:2]

print("Dimension of original image is {}x{}".format(image.shape[1],image.shape[0]))
print("Dimension of template image is {}x{}".format(template.shape[1],template.shape[0]))
print("Dimension of source image is {}x{}\n".format(source.shape[1],source.shape[0]))

plt.subplot(131)
plt.imshow(template,cmap = 'gray'), plt.title("Template Image")
plt.subplot(132)
plt.imshow(source,cmap = 'gray'), plt.title("Source Image")

# Step4
result = ird.similarity(template, source, numiter=3)
assert "timg" in result
plt.subplot(133), plt.imshow(result['timg'],cmap = 'gray'), plt.title("Transformed Image")

print("Results:")
print("Scaling is {}, Rotated Angle is {:.4g}".format(result['scale'], result['angle']))
print("Translation is {}, success rate {:.4g}".format(result['tvec'], result["success"]))
print("Dimension of transformed image is {}x{}\n".format(result['timg'].shape[1],result['timg'].shape[0]))

plt.show()

#Check the results:
# rowsf,colsf = source.shape[:2]
# Mrr = cv2.getRotationMatrix2D((colsf/2,rowsf/2),-24.99,1)
# rott = cv2.warpAffine(source,Mrr,(colsf,rowsf))
# cv2.imshow("Rotate back",rott)
# cv2.waitKey(0)
# Mrtt = np.float32([[1,0,6.73240556],[0,1,-204.36426226]])
# sourcef = cv2.warpAffine(rott,Mrtt,(colsf,rowsf))
# cv2.imshow("Translate back",sourcef)
# cv2.waitKey(0)