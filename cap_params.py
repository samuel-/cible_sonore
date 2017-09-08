import cv2
from time import sleep

def nothing(x):
    pass
def update():
    print "update!"
    for i in l:
        cap.set(i,int(cv2.getTrackbarPos(str(i), 'tracks')*decal[i]))
        sleep(0.1)
    sleep(0.1)
def restore():
    print "restore!"
    for i in l:
        cap.set(i,values[i])

cv2.namedWindow('tracks')
cv2.resizeWindow('tracks',300,850)
cv2.namedWindow('image')
cap = cv2.VideoCapture(0)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920);
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080);
sleep(2)

l=[]
values=[]
decal=[]

for i in range(50):
    val=cap.get(i)
    values.append(val)
    if val != -1:
        #print val
        l.append(i)
        if val>0:
            maxval=int(val*2)
            decal.append(1)
        else:
            maxval=int(-val*2)
            decal.append(-1)
            val=-val
        cv2.createTrackbar(str(i),'tracks',int(val),maxval,nothing)
    else:
        decal.append(0)
print l

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
