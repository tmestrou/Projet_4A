import pandas as pd
import numpy as np
import json
from alive_progress import alive_bar
import time
import cv2
import sys
import os

def ouvrir(lien_dossier):

    liste_fic = os.listdir(lien_dossier)
    
    for i in liste_fic :

        if ('.bmp' not in i) :
            liste_fic.remove(i)
    
    for i in range( len(liste_fic) ) :
        liste_fic[i] = lien_dossier + "/" + liste_fic[i]


    return liste_fic

###################################################

start_time = time.time()

print("\n ##### Programme de quadrillage d'empreintes digitales #####")
heure = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
print("heure :" ,heure, "\n")

with open(sys.argv[1]) as fichierOptions:
    options = json.load(fichierOptions)

print("Récupération des images... \n")


c = 0
l = ouvrir(options['DossierRecupImages'])
if (len(l) == 25) :
    Imag = []

    with alive_bar(5*4, bar='blocks') as bar :
        for i in range(5):
            
            imh =  cv2.imread(l[c])
            c +=1
            
            for j in range(4):
                
                im = cv2.imread(l[c])
                imh = cv2.vconcat([imh,im])
                c += 1
                bar()
                
            Imag.append(imh)

    imv = cv2.hconcat([Imag[0],Imag[1],Imag[2],Imag[3],Imag[4]])

    lien_nv_doss = options['DossierNouvellesImages'] + "/" + options['NomNouvelleImage']
    cv2.imwrite(lien_nv_doss, imv)
    print("\nL'image a bien été créée\n")
else :
    print("Le nombre d'images dans le dossier n'est pas correct")