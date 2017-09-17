import cv2
cap=cv2.VideoCapture(0)
#cap.set(cv2.cv.CV_CAP_PROP_OPENNI_FOCAL_LENGTH,2000)

bright = 50
expose = 20
cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, expose)
#cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE_AUTO,0)

v=cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
w=cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
x=cap.get(cv2.cv.CV_CAP_PROP_GAIN)

print v,w,x


while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    cap.set(cv2.cv.CV_CAP_PROP_BRIGHTNESS, bright)
    cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE, expose)
    v=cap.get(cv2.cv.CV_CAP_PROP_BRIGHTNESS)
    w=cap.get(cv2.cv.CV_CAP_PROP_EXPOSURE)
    print v,w,x
    cv2.imshow('image',img)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
    

      
cap.release()

cv2.destroyAllWindows()
