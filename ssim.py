#from skimage import compare_ssim
import cv2
import numpy as np
from matplotlib import pyplot as plt


def draw_matches(img1, kp1, img2, kp2, matches, color=None): 

    # We're drawing them side by side.  Get dimensions accordingly.
    # Handle both color and grayscale images.
    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))  
    # Place images onto the new image.
    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
    
    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
    r = 15
    thickness = 2
    if color:
        c = color
    for m in matches:
        # Generate random color for RGB/BGR and grayscale images as needed.
        if not color: 
            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
        # wants locs as a tuple of ints.
        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        cv2.line(new_img, end1, end2, c, thickness)
        cv2.circle(new_img, end1, r, c, thickness)
        cv2.circle(new_img, end2, r, c, thickness)
    
    plt.figure(figsize=(15,15))
    plt.imshow(new_img)
    plt.show()
##imageA = cv2.imread(args["first"])
##imageB = cv2.imread(args["second"])
## 
### convert the images to grayscale
##grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
##grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
##
##
##(score, diff) = compare_ssim(grayA, grayB, full=True)
##diff = (diff * 255).astype("uint8")
##print("SSIM: {}".format(score))
##
##
##

cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280);
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720);




cv2.namedWindow('primi')
while cap.isOpened():
    ret, img2 = cap.read()
    if not ret:
        continue
    cv2.imshow('primi', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.imwrite("A.jpg",img2)
while cap.isOpened():
    ret, img2 = cap.read()
    if not ret:
        continue
    cv2.imshow('primi', img2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.imwrite("B.jpg",img2)


cap.release()
cv2.destroyAllWindows()

Original = cv2.imread("A.jpg")

Edited = cv2.imread("B.jpg")

diff = cv2.subtract(Original, Edited)
cv2.imwrite("yo.jpg", diff)


##
##imgray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
##imgray=cv2.equalizeHist(imgray)
##
##ret,thresh = cv2.threshold(imgray,127,255,0)
##contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
##
##cv2.drawContours(Original, contours, -1, (0,255,0), 1)
##cv2.imwrite("see_this.jpg", Original)
##
blank = np.zeros((720,1280,3), np.uint8)


def dodo(imI, imO):
    orb = cv2.ORB()
    #kp = orb.detect(imI,None)
    #kp, des = orb.detectAndCompute(imI, kp)
    kp, des = orb.detectAndCompute(imI, None)
    sortie = cv2.drawKeypoints(imO,kp,color=(0,255,0), flags=0)
    return sortie,kp,des

oriORB,kp1,des1=dodo(Original,blank)
editORB,kp2,des2=dodo(Edited,blank)

diff2 = cv2.subtract(oriORB, editORB)
cv2.imwrite("yorr.jpg", diff2)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
print "j"
print len(matches)

###plt.imshow(blank),plt.show()
##plt.imshow(oriORB),plt.show()
##plt.imshow(editORB),plt.show()
##plt.imshow(diff2),plt.show()



# ratio test as per Lowe's paper
##for i,(m,n) in enumerate(matches):
##    if m.distance < 0.7*n.distance:
##        matchesMask[i]=[1,0]
##    

draw_matches(Original,kp1,Edited,kp2,matches,(250,12,12))





