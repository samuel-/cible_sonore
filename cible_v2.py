# -*- coding: cp1252 -*-

import numpy
import cv2
from threading import Thread
from time import sleep
import sys
#import playsound
#https://pypi.python.org/pypi/playsound/1.2.2


# si
class point(object):
    def __init__(self, coordonnees=(0,0)):
        self.x = coordonnees[0]
        self.y = coordonnees[1]
    def __set__(self, instance, coordonnees):
        self.x = coordonnees[0]
        self.y = coordonnees[1]
        print "point set"
    def __get__(self, instance, owner):
        return (self.x, self.y)

class coords(object) :
    p0 = point()
    p1 = point((0,100))
    p2 = point((100,100))
    p3 = point((100,0))
    def __set__(self, instance, coordonnees):
        self.p0 = (coordonnees[0],coordonnees[1])
        self.p1 = (coordonnees[2],coordonnees[3])
        self.p2 = (coordonnees[4],coordonnees[5])
        self.p3 = (coordonnees[6],coordonnees[7])
    def __get__(self, instance, owner):
        return ([self.p0,self.p1,self.p2,self.p3])

class thread_commande(Thread): 
    def __init__(self):
        Thread.__init__(self)
    def run(self):
        print 'in thread'
        while(True):
            oo=sys.stdin.read(1)
            if oo=="c":
                print "config"
            elif oo=="s":
                print "score"
                jolie_cible.score()
                sleep(4)
            elif oo=="q":
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

class webcam(object) :
    cap = cv2.VideoCapture(0)
    sleep(1)
    def capture(self):
        ret, frame = self.cap.read()
        self.im = frame
        #return frame
    def affiche(self):
        gray = self.im
        #cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
	
    def test(self):
        while(True):
            self.capture()
            self.affiche()
            self.contours()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()
    def contours(self):
        ret,thresh = cv2.threshold(self.im,127,255,0)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        cnt = contours[0]
        M = cv2.moments(cnt)
        print M
        
class cible(object) :
    "cible sarbacane"

    def __init__(self, largeur=1000, hauteur=1000):
        self.l = largeur
        self.h = hauteur
        self.coords = coords()

 
    def score(self):
        prise_ok = False
        for i in range(100) :
            im1 = cam1.capture()
            cam1.affiche()
            im1_new = cam1.capture()
            if (im1==im1_new) :
                prise_ok = True
                break
        if prise_ok :
            print "ye"
            cv2.imshow('fenetre',im1_new)
            #recadrer(im1)
            #homography(im1)
            #recadrer(im2)
            #homography(im2)

        else:
            print 'echec score'

        
        return 'hello world'

    def recadrer(self,image):
        print "recadrage"
        # chercher_cible(image)
     #   image = transform en gris (image)
    #    chercher carré (image)
   #     chercher les coins dans un carré proche des coins
  #      ou chercher les quadrilatères très grands
 #       ou les deux et cross vérifier
#        hop on garde nos coordonnées


jolie_cible = cible(1000,1000)
cam1 = webcam()

#attente clavier dans autre thread !
#> étalonner au début de linstallation de la cible ? (init)
 #   afficher images pour règler grosso merdo
#    l'idée cest de chercher les coins dans un carré proche des coins
#>cible.score

commande = thread_commande()
commande.start()
commande.join()
