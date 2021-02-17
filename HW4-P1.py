import cv2
import numpy as np
from matplotlib import pyplot as plt
from math import copysign, log10

#Import Image, Gray scale, Thresholding
imageH = cv2.imread('Figures/HCharacter.png')
imageI = cv2.imread('Figures/ICharacter.png')
imageT = cv2.imread('Figures/TCharacter.png')
imageH_Gray = cv2.cvtColor(imageH, cv2.COLOR_BGR2GRAY)
imageI_Gray = cv2.cvtColor(imageI, cv2.COLOR_BGR2GRAY)
imageT_Gray = cv2.cvtColor(imageT, cv2.COLOR_BGR2GRAY)
retH,Hthresh = cv2.threshold(imageH_Gray, 5,255,cv2.THRESH_BINARY)
retI,Ithresh = cv2.threshold(imageI_Gray,250,255,cv2.THRESH_BINARY_INV)
retT,Tthresh = cv2.threshold(imageT_Gray,250,255,cv2.THRESH_BINARY_INV)
plt.subplot(321), plt.imshow(imageH[:,:,::-1]), plt.title("H-Original")
plt.subplot(323), plt.imshow(imageI[:,:,::-1]), plt.title("I-Original")
plt.subplot(325), plt.imshow(imageT[:,:,::-1]), plt.title("T-Original")

#Check
#cv2.imshow('image',imageI)
#cv2.imshow('gray',imageI_Gray)
#cv2.imshow('thresh',Tthresh)

#Step1: H
plt.subplot(322), plt.imshow(Hthresh,'gray'), plt.title("H-Character Image")
H_moments = cv2.moments(Hthresh)
H_huMoments = cv2.HuMoments(H_moments)
H_loghuMoments = np.zeros((7,1))
for i in range(0,7):
    H_loghuMoments[i] = -1* copysign(1.0, H_huMoments[i]) * log10(abs(H_huMoments[i]))
    print("I_H[",i,"] = ",H_loghuMoments[i])
print("\n")

#Step1: I
plt.subplot(324), plt.imshow(Ithresh,'gray'), plt.title("I-Character Image")
I_moments = cv2.moments(Ithresh)
I_huMoments = cv2.HuMoments(I_moments)
I_loghuMoments = np.zeros((7,1))
for i in range(0,7):
    I_loghuMoments[i] = -1* copysign(1.0, I_huMoments[i]) * log10(abs(I_huMoments[i]))
    print("I_I[",i,"] = ",I_loghuMoments[i])
print("\n")

#Step1: T
plt.subplot(326), plt.imshow(Tthresh,'gray'), plt.title("T-Character Image")
T_moments = cv2.moments(Tthresh)
T_huMoments = cv2.HuMoments(T_moments)
T_loghuMoments = np.zeros((7,1))
for i in range(0,7):
    T_loghuMoments[i] = -1* copysign(1.0, T_huMoments[i]) * log10(abs(T_huMoments[i]))
    print("I_T[",i,"] = ",T_loghuMoments[i])
print("\n")

plt.show()