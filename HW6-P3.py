import numpy as np 
import cv2         
from matplotlib import pyplot as plt

# Step 1
image = cv2.imread('Figures\WalkingMotionFrame.png')
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.subplot(121), plt.imshow(image_gray,cmap='gray'), plt.title("Grayscale Image")
# Step 2
feature_params=dict( maxCorners=60, qualityLevel=0.25, minDistance=10, blockSize=10 )

# Step 3
ptcorner = cv2.goodFeaturesToTrack(image_gray, mask = None, **feature_params)
ptcorner = np.int0(ptcorner)
for i in ptcorner:
    x,y = i.ravel()  
    cv2.circle(image,(x,y),3,(0,0,255),-1)
plt.subplot(122), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Corner points in Frame1")
plt.show()

# Step 4
lk_params=dict( winSize=(15,15), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
cap = cv2.VideoCapture('Figures\WalkingMotion.mp4')
ret, first_frame = cap.read()
old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
mask = np.zeros_like(first_frame)

while(cap.isOpened()):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1[st==1] 
    good_old = p0[st==1] 
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel() 
        c,d = old.ravel() 
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2) 
        frame = cv2.circle(frame,(a,b),3,color[i].tolist(),-1)  
    img = cv2.add(frame,mask) 
    cv2.imshow("sparse optical flow",img) 
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    
cv2.destroyAllWindows()
cap.release()









