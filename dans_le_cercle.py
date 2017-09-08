
import cv2
import numpy as np
#from math import sqrt

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

def color_mouse(event,x,y,flags,param):
    global img, mouse, color
    if event == cv2.EVENT_MOUSEMOVE:
        (mouse[0],mouse[1])=(x,y)
        if inside_circle(c,x,y):
            color=(0,0,255)
        else:
            color=(255,0,0)
    if event == cv2.EVENT_LBUTTONUP:
        c.set_center(x,y)


def inside_circle(c,x,y):
    if pow(x-c.center[0],2)+pow(y-c.center[1],2)>pow(c.radius,2):
        return True
    else:
        return False

cap = cv2.VideoCapture(0)
cv2.namedWindow('jeu')
cv2.setMouseCallback('jeu',color_mouse)
c=cercle()
color=(255,0,0)
mouse=[0,0]
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    cv2.line(img,(mouse[0],mouse[1]-15),(mouse[0],mouse[1]+15), color, 2)
    cv2.line(img,(mouse[0]-15,mouse[1]),(mouse[0]+15,mouse[1]), color, 2)
    cv2.circle(img,(c.center[0],c.center[1]), c.radius, (0,255,0), 2)

    cv2.imshow('jeu',img)
    key = cv2.waitKey(33)
    if key == ord('+'):
        c.radius_up(5)
    elif key == ord('-'):
        c.radius_down(5)
    elif key == ord('q'):
        break
            
cap.release()
cv2.destroyAllWindows()








cv2.destroyAllWindows()
