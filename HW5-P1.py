import numpy as np 
import cv2         
import argparse    
import imutils     
import glob2 

# Argument Parser
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

# Step 1
template = cv2.imread(args["template"]) 
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Step 2
template = cv2.Canny(template, 50, 200) 
(hT, wT) = template.shape[:2] 
cv2.imshow("Car Template", template)
cv2.waitKey(0)

# Step 3
for imagePath in glob2.glob(args["images"] + "/*.jpg"):
    print(imagePath)
    image = cv2.imread(imagePath) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert TemplateImage to grayscale
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        # resize the image according to the scale, and keep track of the ratio of the resizing
        ResizedImage = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(ResizedImage.shape[1])
        # if the resized image is smaller than the template, then break from the loop
        if (ResizedImage.shape[0] < hT) or (ResizedImage.shape[1] < wT):
            break
        edged = cv2.Canny(ResizedImage, 50, 200) # detect edges in resized, grayscale image using "Canny" algorithm
        # apply template matching to find the template in the image
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result) # find maxVal and its location
        print("Scale = ",1.0/r, "; MaxValue = ",maxVal)
        # check to see if the iteration should be visualized
        if args.get("visualize", False):
            # draw a bounding box around the detected region
            clone = np.dstack([edged, edged, edged])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),(maxLoc[0]+wT, maxLoc[1]+hT), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)
        # if we have found a new maximum correlation value, then update the bookkeeping variable
        if (found is None) or (maxVal > found[0]):
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates of the bounding box based on the resized ratio
    (maxVal, maxLoc, r) = found
    print("Best Match Scale = ",1.0/r,"; MaxValue = ", maxVal)
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + wT) * r), int((maxLoc[1] + hT) * r))
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2) # draw a bounding box around the detected result
    cv2.imshow("Image: "+imagePath, image)
    cv2.waitKey(0)
    cv2.destroyWindow("Visualize")
cv2.waitKey(0)
cv2.destroyAllWindows()
    
#CMD: python HW5-P1.py --template Figures/CarTemplate.jpg --images Figures/Cars --visualize 1
