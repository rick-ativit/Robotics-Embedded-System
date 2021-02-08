import cv2      
import numpy as np 
from matplotlib import pyplot as plt

# Part a: Step 1
image = cv2.imread('Figures/ACharacter.png')
rows,cols = image.shape[:2]
print("width=",cols," height=",rows)
Mr = cv2.getRotationMatrix2D((cols/2,rows/2),-90,1)
OutArray1 = cv2.warpAffine(image,Mr,(cols,rows))
rows1,cols1 = OutArray1.shape[:2]
print("width=",cols1," height=",rows1)
# Part a : Step 2
Mt = np.float32([[1,0,50],[0,1,100]])
OutArray1 = cv2.warpAffine(OutArray1,Mt,(cols+200,rows+200))
cv2.imwrite('Figures/ACharacter_Parta_v1.jpg',OutArray1)

# Part b : Step 1
Mt = np.float32([[1,0,50],[0,1,100]])
OutArray2 = cv2.warpAffine(image,Mt,(cols+200,rows+200))
# Part b : Step 2
Mr = cv2.getRotationMatrix2D((cols/2+50,rows/2+100),-90,1)
OutArray2 = cv2.warpAffine(OutArray2,Mr,(cols+200,rows+200))
cv2.imwrite('Figures/ACharacter_Partb_v1.jpg',OutArray2)

#Show all image
plt.subplot(211), plt.imshow(OutArray1[:,:,::-1])
plt.subplot(212), plt.imshow(OutArray2[:,:,::-1])
plt.show()

pic1 = cv2.imread("Figures/ACharacter_Parta_v1.jpg")
pic2 = cv2.imread("Figures/ACharacter_Partb_v1.jpg")
difference = cv2.subtract(pic1, pic2)    
result = not np.any(difference)
if result is True:
    print("Pictures are identical")
else:
    print("Pictures are not identical")

