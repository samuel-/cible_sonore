import cv2
from time import sleep

def nothing(i):
    pass

cv2.namedWindow('image')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920);
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080);
sleep(2)

for i in range(50):
    val=cap.get(i)
    if val != -1:
        print val
        cv2.createTrackbar(str(i),'image',0,1255,nothing,i)

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue
    cv2.imshow('image',img)
    


    key = cv2.waitKey(33)
    if key == ord('+') :
        print "..."
        cap.set(cv2.cv.CV_CAP_PROP_FOCUS,40);
        print "done"
    elif key == ord('-') :
        print "restore"
        cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE,20);
        print "done"
    elif key == 32 :
        update()
    elif key == 2555904 or key == 2424832: # fleches droite gauche
        nothing()

        
    elif key == ord('q'):
        break




        
cap.release()
cv2.destroyAllWindows()
