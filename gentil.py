import numpy as np
import cv2
from time import sleep, clock
from math import pow, atan
#import fitEllipse
from playsound import playsound
from matplotlib import pyplot as plt
#https://stackoverflow.com/questions/19375675/simple-way-of-fusing-a-few-close-points


def nothing():
    pass

##def draw_matches(img1, kp1, img2, kp2, matches, color=None): 
##
##    # We're drawing them side by side.  Get dimensions accordingly.
##    # Handle both color and grayscale images.
##    if len(img1.shape) == 3:
##        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], img1.shape[2])
##    elif len(img1.shape) == 2:
##        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1])
##    new_img = np.zeros(new_shape, type(img1.flat[0]))  
##    # Place images onto the new image.
##    new_img[0:img1.shape[0],0:img1.shape[1]] = img1
##    new_img[0:img2.shape[0],img1.shape[1]:img1.shape[1]+img2.shape[1]] = img2
##    
##    # Draw lines between matches.  Make sure to offset kp coords in second image appropriately.
##    r = 15
##    thickness = 2
##    if color:
##        c = color
##    for m in matches:
##        # Generate random color for RGB/BGR and grayscale images as needed.
##        if not color: 
##            c = np.random.randint(0,256,3) if len(img1.shape) == 3 else np.random.randint(0,256)
##        # So the keypoint locs are stored as a tuple of floats.  cv2.line(), like most other things,
##        # wants locs as a tuple of ints.
##        end1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
##        end2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
##        cv2.line(new_img, end1, end2, c, thickness)
##        cv2.circle(new_img, end1, r, c, thickness)
##        cv2.circle(new_img, end2, r, c, thickness)
##    
##    plt.figure(figsize=(15,15))
##    plt.imshow(new_img)
##    plt.show()

def get_coords(event,x,y,flags,param):
    global cadre_souris,pts_cadre
    global recadre
    if x+CROPSQUARE>img_w : x=img_w-CROPSQUARE
    if y+CROPSQUARE>img_h : y=img_h-CROPSQUARE
    if x-CROPSQUARE<0 : x=CROPSQUARE
    if y-CROPSQUARE<0 : y=CROPSQUARE
    if event == cv2.EVENT_MOUSEMOVE:
        (cadre_souris[0],cadre_souris[1]) = (x-CROPSQUARE,y-CROPSQUARE)
        (cadre_souris[2],cadre_souris[3]) = (cadre_souris[0]+DOBLSQUARE,cadre_souris[1]+DOBLSQUARE)
    if event == cv2.EVENT_LBUTTONUP:
        print "click"
        for i in range(4):
            if pts_cadre[i][0]==-1:
                break
        if i!=3:
            (pts_cadre[i][0],pts_cadre[i][1]) = (x,y)
        elif i==3 and recadre==False:
            (pts_cadre[i][0],pts_cadre[i][1]) = (x,y)
            recadre=True
            valid_cadre()
            print "recadre!"

def get_circles(event,x,y,flags,param):
    #trouver les cercles ?
    global souris_x,souris_y
    global cercles, c_i
    if event == cv2.EVENT_MOUSEMOVE and cible == False:
        souris_x=x
        souris_y=y
        cercles[c_i].set_center(x,y)
    if event == cv2.EVENT_LBUTTONUP and cible == False:
        print "next"
        if c_i<4:
            cercles.append(cercle(cercles[c_i].radius+10,c_i+1))
            c_i+=1
            #order_cercles(cercles)
    if event == cv2.EVENT_RBUTTONUP and cible == False:
        print "previous"
        if c_i>0:
            c_i-=1
            cercles.pop()

def affiche_corners(image):
    for i in range(4):
        if pts_cadre[i][0]!=-1:
            cv2.circle(image,(pts_cadre[i][0],pts_cadre[i][1]), 13, (0,0,255), 1)
            cv2.line(image,(pts_cadre[i][0],pts_cadre[i][1]-5),(pts_cadre[i][0],pts_cadre[i][1]+5), (0,0,255), 1)
            cv2.line(image,(pts_cadre[i][0]-5,pts_cadre[i][1]),(pts_cadre[i][0]+5,pts_cadre[i][1]), (0,0,255), 1)

def affiche_fleches(image):
    for i,f in enumerate(fleches):
        (x,y)=int(f.pt[0]),int(f.pt[1])
        #cv2.circle(image,(x,y), 13, (0,0,255), 1)
        cv2.line(image,(x,y-5),(x,y+5), (0,0,255), 1)
        cv2.line(image,(x-5,y),(x+5,y), (0,0,255), 1)

def order_points(pts):
    # top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def transform(image, points, w, h):
    rect=points
    dst = np.float32([[0,0],[w,0],[w,h],[0,h]])
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

class cercle(object):
    def __init__(self, radius=25,color=0):
        self.center = (0,0)
        self.radius = radius
        self.color = color
    def set_center(self,x,y):
        self.center = (x,y)
    def radius_up(self,inc):
        if self.radius<250:
            self.radius += inc
    def radius_down(self,inc):
        if self.radius>5:
            self.radius -= inc

class fleche(object):
    def __init__(self, pt=(1,1), color=0):
        self.pt = pt
        self.color = color
        self.score = score_fleche(pt[0],pt[1])
        self.time = clock()
    def set_center(self,x,y):
        self.center = (x,y)

def valid_cadre():
    global pts_cadre_o,pts_lines
    pts = np.array(pts_cadre, np.float32)
    pts_cadre_o = order_points(pts)
    pts_lines=pts_cadre_o.astype(int).reshape((-1,1,2))
    cv2.namedWindow('destination')
    cv2.setMouseCallback('destination',get_circles)

def valid_cible():
    global cercles
    cible=True
    cercles.pop()
    cx_list=[]
    cy_list=[]
    for i,c in enumerate(cercles):
        cx_list.append(c.center[0])
        cy_list.append(c.center[1])
    cx=int(round(float(sum(cx_list))/float(len(cx_list))))
    cy=int(round(float(sum(cy_list))/float(len(cy_list))))
    for _,c in enumerate(cercles):
        c.center=(cx,cy)
    cv2.destroyAllWindows()

def recadrer(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if recadre==False:
        if mode_gray == False:
            cv2.rectangle(img, (cadre_souris[0],cadre_souris[1]), (cadre_souris[2],cadre_souris[3]), (50, 255, 50), 2)
            cv2.line(img, (cadre_souris[0],cadre_souris[1]), (cadre_souris[2],cadre_souris[3]), (0, 250, 0), 1)
            cv2.line(img, (cadre_souris[2],cadre_souris[1]), (cadre_souris[0],cadre_souris[3]), (0, 250, 0), 1)
            affiche_corners(img)
            cv2.imshow('primi', img)
        else:
            carre = gray[cadre_souris[1]:cadre_souris[3], cadre_souris[0]:cadre_souris[2]]
            equ = cv2.equalizeHist(carre)
            gray[cadre_souris[1]:cadre_souris[3], cadre_souris[0]:cadre_souris[2]] = equ
            affiche_corners(gray)
            cv2.imshow('primi', gray)
    else:
        warp = transform(img,pts_cadre_o,600,600)
        if mode_gray == False:
            cv2.polylines(img,[pts_lines],True,(0,255,255),1)
            cv2.imshow('primi', img)
        else:
            cv2.polylines(gray,[pts_lines],True,(0,255,255))
            cv2.imshow('primi', gray)
        if cible==False:
            cv2.line(warp,(souris_x,souris_y-5),(souris_x,souris_y+5), (0,0,255), 2)
            cv2.line(warp,(souris_x-5,souris_y),(souris_x+5,souris_y), (0,0,255), 2)
            for c in cercles:
                cv2.circle(warp,c.center,c.radius,colors[c.color],2)
        cv2.imshow('destination',warp)
        cv2.imwrite("warp.png",warp)

def dodo(imI, imO, n_kp=100):
    orb = cv2.ORB()
    orb.setInt("nFeatures", n_kp)
    kp = orb.detect(imI,None)
    #for i in range(len(kp)):
    kp = fuse(kp,50)
    kp, des = orb.compute(imI, kp)
    #kp, des = orb.detectAndCompute(imI, None)
    sortie = cv2.drawKeypoints(imO,kp,color=(0,255,0), flags=0)
    return sortie,kp,des

def inside_circle(x,y,c):
    if pow(x-c.center[0],2)+pow(y-c.center[1],2)<pow(c.radius,2):
        return True
    else:
        return False
    
def score_fleche(x,y):
    scores=[100,50,30,20,10,5]
    score=0
    for i,c in enumerate(cercles):
        if inside_circle(x,y,c):
            score=scores[i]
            break
    return score

def angle_fleche(x,y):
    (dx,dy) = (x-cercles[0].center[0], y-cercles[0].center[1])
    if dx==0:
        dx=0.0001
    return atan(float(y)/float(x))

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse(kp, d):
    ret = []
    d2 = d * d
    n = len(kp)
    taken = [False] * n
    for i in range(n):
        if not taken[i]:
################
            count = 1
            new_size = kp[i].size
            new_kp = kp[i]
            new_x,new_y = new_kp.pt[0]*new_size, new_kp.pt[1]*new_size
            taken[i] = True
            for j in range(i+1, n):
                if dist2(kp[i].pt, kp[j].pt) < d2:
                    new_x += kp[j].pt[0]*kp[j].size
                    new_y += kp[j].pt[1]*kp[j].size
                    count += 1
                    new_size += kp[j].size
                    taken[j] = True
            new_x /= float(new_size)
            new_y /= float(new_size)
            new_kp.pt = (new_x,new_y)
            new_kp.size = new_size / float(count)
##################            
##            count = 1.
##            new_kp = kp[i]
##            new_x,new_y = new_kp.pt
##            taken[i] = True
##            for j in range(i+1, n):
##                if dist2(kp[i].pt, kp[j].pt) < d2:
##                    new_x += kp[j].pt[0]
##                    new_y += kp[j].pt[1]
##                    count += 1
##                    taken[j] = True
##            new_x /= count
##            new_y /= count
##            new_kp.pt = (new_x,new_y)
##################
            ret.append(new_kp)
    return ret


def extract(kp1,kp2,d=50):
    #kp1=fuse(kp1,d)
    #kp2=fuse(kp2,d)
    kp=kp1+kp2
    ret = []
    d2 = d * d 
    n = len(kp2)
    for i,k2 in enumerate(kp2):
        keep_it = True
        for j,k1 in enumerate(kp1):
            if dist2(k2.pt, k1.pt) < d2:
                keep_it = False
        if keep_it :
            ret.append(kp[i])
    if ret==[]:
        print "nvlle fleche non trouvee"
        return False
    elif len(ret)>1:
        print "trouve trop de points pour la nouvelle fleche"
        return False
    elif len(ret)==1:
        print ret[0].pt
        return ret[0].pt

def tableau():
    global fleches
    score_tot = 0
    for f,fleche in enumerate(fleches):
        print f,fleche.score,fleche.pt
        score_tot += fleche.score
    print "score total"
    print score_tot

def jouer(img):
    global fleches
    global kplen0
    #oldwarp=None
    warp = transform(img,pts_cadre_o,600,600)
    oldwarp=warp
    _,kp_0,_=dodo(warp,warp)
    kplen0 = len(kp_0)
    mode=3
    timer=0
    while True:
        if len(fleches)==0 or (clock()-fleches[-1].time)>3:
            oldwarp=warp
        #cv2.imwrite("A.jpg",oldwarp)
        ret=False
        sleep(0.5)
        while not ret:
            ret, img2 = cap.read()
        warp = transform(img2,pts_cadre_o,600,600)
        #cv2.imwrite("B.jpg",warp)
        Original = oldwarp
        Edited = warp
        #blank = np.zeros((720,1280,3), np.uint8)
        #blank[:,:] = (255,255,255)
        orb1,kp1,des1=dodo(Original,Original)
        orb2,kp2,des2=dodo(Edited,Edited)
        # ?
        print "len kp1 ", len(kp1)
        print "len kp2 ", len(kp2)
        if len(kp1)<kplen0 or len(kp2)<kplen0:
            continue
        if len(kp1)>kplen0:
            timer=clock()
        if len(kp1)>kplen0 and (clock()-timer)>4 and (clock()-timer)<7:
            kplen0=len(kp1) ##.....hmm...
        if len(kp1)==kplen0 and len(kp2)==kplen0+1:
            print "yeeeeeeha !?"
            playsound('sons/velo.wav')
            fl = extract(kp1,kp2)
            if fl != False:
                fleches.append(fleche(fl))
                tableau()
                oldwarp=warp
                kplen0 += 1
                
        cv2.imwrite("orb1.jpg", orb1)
        cv2.imwrite("orb2.jpg", orb2)
        #for c in cercles:
            #cv2.circle(warp,c.center,c.radius,colors[c.color],1)
        if mode == 0:
            cv2.imshow('jeu',orb2)
        elif mode == 1:
            cv2.imshow('jeu',Edited)
        elif mode == 2:
            for c in cercles:
                cv2.circle(Edited,c.center,c.radius,colors[c.color],1)
            cv2.imshow('jeu',Edited)
        elif mode == 3:
            for c in cercles:
                cv2.circle(orb2,c.center,c.radius,colors[c.color],1)
            cv2.imshow('jeu',orb2)
            
        key = cv2.waitKey(33)
        if key == 2555904: # fleche droite 
            if mode<3: mode +=1
            else: mode=0
        elif key == 2424832: # fleche gauche
            if mode>0: mode -=1
            else: mode=3
        elif key == ord('q'):
            break

    return kp1,kp2
    #draw_matches(Original,kp1,Edited,kp2,matches,(250,12,12))


  
def jouer_old(img):
    warp = transform(img,pts_cadre_o,600,600)
    while True:
        ret, img2 = cap.read()
        if not ret:
            continue
        warp = transform(img2,pts_cadre_o,600,600)
        cv2.imshow('jeu',warp)
        if cv2.waitKey(1) == ord('s'):
            break
    cv2.imwrite("A.jpg",warp)
    while True:
        ret, img2 = cap.read()
        if not ret:
            continue
        warp = transform(img2,pts_cadre_o,600,600)
        cv2.imshow('jeu',warp)
        if cv2.waitKey(1) == ord('s'):
            break
    cv2.imwrite("B.jpg",warp)
    Original = cv2.imread("A.jpg")
    Edited = cv2.imread("B.jpg")
    blank = np.zeros((720,1280,3), np.uint8)
    blank[:,:] = (255,255,255)
    orb1,kp1,des1=dodo(Original,Original)
    orb2,kp2,des2=dodo(Edited,Edited)
    #print len(des1)
    #print len(des2)
    cv2.imwrite("orb1.jpg", orb1)
    cv2.imwrite("orb2.jpg", orb2)
    diff2 = cv2.subtract(orb1, orb2)
    cv2.imwrite("orb_diff.jpg", diff2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    print "j"
    print len(matches)
    #for c in cercles:
     #   cv2.circle(warp,c.center,c.radius,colors[c.color],1)
    cv2.imshow('jeu',warp)
    return kp1,kp2
    #draw_matches(Original,kp1,Edited,kp2,matches,(250,12,12))
    
def ouvrir_camera():
    global img_h, img_w, img_chs
    global cible, mode_gray, action
    #1920x1080
    #1600x900
    #1366x768
    #1280x720
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 1080);
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            continue
        #cv2.imshow('primi', img)
        img_h, img_w, img_chs = img.shape
        #print img_h
        #print img_w
        if action=='recadrer':
           recadrer(img)
        elif action=='jouer':
            
            kp1,kp2=jouer(img)
            break

        key = cv2.waitKey(33)
        if key == ord('+') and cible==False:
            cercles[c_i].radius_up(5)
        elif key == ord('-') and cible==False:
            cercles[c_i].radius_down(5)
        elif key == 32 and cible==False: # espace
            print "miiii"
            valid_cible()
            action='jouer'
        elif key == 2555904 or key == 2424832: # fleches droite gauche
            mode_gray = not mode_gray
        elif key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return img,kp1,kp2

img_h=1
img_w=1
img_chss=1
CROPSQUARE = 30
DOBLSQUARE = 2*CROPSQUARE
cadre_souris=[-1,-1,-1,-1]
pts_cadre=[[-1,-1],[-1,-1],[-1,-1],[-1,-1]]
pts_cadre_o = None
#np.zeros((4, 2), dtype = "float32")
pts_lines = None
fleches=[]
kplen0=0
cercles=[cercle()]
c_i=0
colors=[(250,22,122),(22,122,250),(122,250,22),(250,122,22),(122,22,250),(22,250,122)]
########
souris_x=1
souris_y=1
########
recadre=False
cible=False
ellipse=False
mode_gray = False
action='recadrer'
####################################
cap = cv2.VideoCapture(0)
cv2.namedWindow('primi')
cv2.setMouseCallback('primi',get_coords)

img2,kp1,kp2=ouvrir_camera()

cv2.imwrite("img.png",img2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("img_g.png",warp)
