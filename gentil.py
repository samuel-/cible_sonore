import numpy as np
import cv2
from time import sleep
#import math
#import fitEllipse
from playsound import playsound

def nothing():
    pass

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

def valid_cadre():
    global pts_cadre_o,pts_lines
    pts = np.array(pts_cadre, np.float32)
    pts_cadre_o = order_points(pts)
    pts_lines=pts_cadre_o.astype(int).reshape((-1,1,2))
    cv2.namedWindow('destination')
    cv2.setMouseCallback('destination',get_circles)

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

def jouer(img):
    warp = transform(img,pts_cadre_o,600,600)
    for c in cercles:
        cv2.circle(warp,c.center,c.radius,colors[c.color],1)
    cv2.imshow('jeu',warp)

    
def ouvrir_camera():
    global img_h, img_w, img_chs
    global cible, mode_gray, action
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 720);
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
            jouer(img)

        key = cv2.waitKey(33)
        if key == ord('+') and cible==False:
            cercles[c_i].radius_up(5)
        elif key == ord('-') and cible==False:
            cercles[c_i].radius_down(5)
        elif key == 32 and cible==False: # espace
            print "miiii"
            cible=True
            cercles.pop()
            cv2.destroyAllWindows()
            action='jouer'
        elif key == 2555904 or key == 2424832: # fleches droite gauche
            mode_gray = not mode_gray
        elif key == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    return img

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
cercles=[cercle()]
c_i=0
colors=[(250,22,122),(22,122,250),(122,250,22),(250,122,22),(122,22,250),(22,250,122)]
souris_x=1
souris_y=1
recadre=False
cible=False
ellipse=False
mode_gray = True
action='recadrer'
####################################
cap = cv2.VideoCapture(0)
cv2.namedWindow('primi')
cv2.setMouseCallback('primi',get_coords)

img2=ouvrir_camera()

cv2.imwrite("img.png",img2)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#cv2.imwrite("img_g.png",warp)
