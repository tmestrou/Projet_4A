# Packages à installer :
# numpy - matplotlib.pyplot - pandas - random - PIL - scikit-learn - scipy - opencv-python - sys - cv2 - json - 
# time - alive-progress - rotate_matrix

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from PIL import Image
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os
import sys
import cv2
import json
import time
import rotate_matrix
from alive_progress import alive_bar

def ouvrir(lien_dossier, l_im, h_im, n_axe):
        
    # On lit la liste des fichiers présents dans le dossier
    liste_fic = os.listdir(lien_dossier)
    for i in liste_fic :
        # On supprime tout les fichiers ne correspondant pas au format d'image que l'on souhaite étudier
        if '.bmp' not in i :
            liste_fic.remove(i)

    # On récupère le nombre d'images et de composants
    n_im = len(liste_fic)

    # On créé un array de dimension 2 qui contiendra toutes les informations de toutes les images
    M = np.zeros((n_im, l_im*h_im))
    

    #for k in progressbar(range(n_im), "Récupération des images: ", 40):
    with alive_bar(n_im, bar='blocks') as bar:
        ind = 0

        # On lit les informations de chaque image et les implémente dans M
        for i in liste_fic :
            chaine = lien_dossier + '/' + i
            img = Image.open(chaine)
            M[ind] = (np.array(img.getdata()).reshape(1,h_im*l_im))
            ind += 1
            bar()

    # On créé un dataframe contenant les informations des images
    df = pd.DataFrame(M)

    # On applique la PCA aux données des images pour obtenir un dataframe ne contenant que les axes principaux pour chaque image
    pca = PCA(n_components = n_axe)
    df_proj = pd.DataFrame(pca.fit_transform(df))

    return df, df_proj, pca



def save(nom_fichier, lien_dossier, vec_image, afficher) :
    path = lien_dossier + '/' + nom_fichier + '.bmp'
    
    cv2.imwrite(path,vec_image)
    
    if (afficher) :
        im = Image.fromarray(np.reshape(vec_image,(h_im,l_im)))
        im.show()

    
    
# Prend en paramètre un dataframe et renvoie un dataframe
def cha(data) :
    
    # On calcule les distance de chaque image par rapport aux autres
    proj_cha = linkage(data,method="ward",metric="euclidean")

    # On calcule le dendrogramme associé
    d = dendrogram(proj_cha,orientation="top")
    
    # On récupère les hauteurs des groupes qu'a formé le dendrogramme
    coord = np.array(d['dcoord'])
    coord = coord[coord != 0] # On supprime les hauteurs nulles 
    m = coord.mean()
    s = coord.std()

    dist = coord.mean() + np.random.randint(-s, s)
    
    grpe = fcluster(proj_cha,t=dist,criterion='distance')
    
    # On ajoute le numéro de groupe de la CHA aux images
    data['cha'] = grpe
    
    # On choisit un groupe au hasard parmis ceux qui ont été créés lors de la CHA
    n_groupe = random.randint(1, max(grpe))
    
    # On renvoie les informations des images du groupe sélectionné sans la colonne "cha"
    return (data[data['cha'] == n_groupe].drop(columns='cha'))


# Fonction qui prend en paramètre un dataframe d'axes d'images et qui renvoie un array contenant les données de la nouvelle image
def moyenne_axe(data):
    # On récupère les informations du dataframe
    n_im = data.shape[0]
    n_compo = data.shape[1]

    if (n_im != 1) : # s'il y a plus d'une image, on va traiter le dataframe
        data_traite = np.zeros(n_compo)
        for i in range (0, n_compo): # On étudie chaque axe pour toutes les images
            compo_i = np.array(data.iloc[:, i]) # compo_i est un tableau contenant les informations de l'axe i de toutes les images

            Xcentre = np.random.randint(0, n_im) # On prend une image au hasard parmi celles du dataframe

            Ycentre = compo_i[Xcentre] # On sélectionne la valeur de l'axe i correspondant à l'image sélectionnée au hasard

            v = np.std(compo_i) # On calcule l'écart-type de l'axe i

            # On sélectionne les valeurs de l'axe i proches de Ycentre + ou - l'écart-type de l'axe i et on renvoie la moyenne de ces valeurs
            data_traite[i] = compo_i[(compo_i > Ycentre-v) & (compo_i < Ycentre+v)].mean()
    else :
        data_traite = np.array(data)
        
    return data_traite

def bruitage_gaussienne(data, n_axe):
    U = []
    x = np.random.randint(data.shape[0])
    for i in range(0,n_axe):
        U.append(abs(min(data[i])) + abs(max(data[i])))

    cov = np.eye(n_axe) * np.sqrt(U)
    mean = np.zeros(n_axe)

    V = n_axe * np.random.multivariate_normal(mean,cov)

    proj = data.iloc[x] + V

    return proj


def tableau_multi_images(data, i, j , X, pca) :

    M = np.zeros((5,5))
    M[0] = np.array([1,0.75,0.5,0.25,0])
    M[1] = np.array([0.75,3/6,3/8,1/6,0])
    M[2] = np.array([0.5,3/8,0.25,1/8,0])
    M[3] = np.array([0.25,1/6,1/8,1/6,0])

    N = rotate_matrix.anti_clockwise(M)

    P = rotate_matrix.anti_clockwise(N)

    O = rotate_matrix.anti_clockwise(P)

    proj = M[i][j] * data.iloc[X[0], :] + N[i][j] * data.iloc[X[1], :] + O[i][j] * data.iloc[X[2], :] + P[i][j] * data.iloc[X[3], :]
    proj2 = pca.inverse_transform(proj)

    return np.reshape(proj2,(h_im,l_im))

def traitement(data, pca, l_im, h_im, n_axe, typ):
    n_compo = data.shape[1]
    vecteur_traite = np.zeros(n_compo)

    # Moyenne évoluée par axe
    if typ == 1 :
        vecteur_traite = moyenne_axe(data)
    
    # Moyenne brute des axes
    elif typ == 2 :
        vecteur_traite = np.array(data.mean(axis=0))
    
    # CHA puis moyenne brute
    elif typ == 3:
        data = cha(data)
        vecteur_traite = data.mean()

    # CHA puis moyenne plus évoluée
    elif typ == 4 :
        data = cha(data)
        #print(data)
        vecteur_traite = moyenne_axe(data)  
        
    # Rajout de bruit selon une gaussienne
    elif typ == 5 :
        vecteur_traite = bruitage_gaussienne(data, n_axe)
    
    data2 = pca.inverse_transform(vecteur_traite)
    return np.reshape(data2,(h_im,l_im)) 


def score(img1, img2, pasH, pasL):
    score = 0
    h = img1.shape[0]
    l = img1.shape[1]
    nbCaseH = int( h / pasH )
    nbCaseL = int( l / pasL )
    
    for i in range (nbCaseH):
        for j in range (nbCaseL):
            meanImg1 = np.mean(img1[ i * pasH:i * pasH + pasH - 1, j * pasL:j * pasL + pasL - 1] )
            meanImg2 = np.mean(img2[ i * pasH:i * pasH + pasH - 1, j * pasL:j * pasL + pasL - 1] )
            
            #distance entre les deux carrés
            #score+=(meanImg1-meanImg2)**2
            
            #différence des moyennes entre les deux carrés
            score += abs(meanImg1 - meanImg2)
    score /= (nbCaseH * nbCaseL * 255)
    
    return score

# Fonction qui va comparer la nouv_im à toutes les images du dataframe df_anc_im
def compare(nouv_im, df_anc_im) :
    sco = []
    # Pour chaque image dans le dataframe, on calcule le score avec la nouvelle image et on l'ajoute au tableau sco
    for i in range (len(df_anc_im)) :
        anc_im = np.array(df_anc_im.iloc[i,:]).reshape(h_im, l_im)
        #i_array = np.array(anc_im).
        sco.append(score(nouv_im, anc_im, 50, 50))
    return sco


def progressbar(it, prefix = "", size = 60, file = sys.stdout):
    count = len(it)

    def show(j):
        x = int(size * j / count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#" * x, "." * (size - x), j, count))
        file.flush()
        file.write("\n")

    for i, item in enumerate(it):
        yield item
        show(i + 1)
        file.write("\n")
    file.flush()

def type(typ) :
    if (typ == 1) :
        return "Moyenne brute par axe de l'ACP"
    elif (typ == 2) :
        return "Moyenne évoluée par axe de l'ACP"
    elif (typ == 3) :
        return "CAH puis moyenne brute des axes"
    elif (typ == 4) :
        return "CAH puis moyenne évoluée des axes"
    elif (typ == 5) :
        return "Bruit gaussien"
    elif (typ == 6) :
        return "Tableau multi-image"




################## Programme principal ##################

start_time = time.time()

print("\n ##### Programme de recalage d'empreintes digitales #####")
heure = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
print("heure :" ,heure, "\n")

with open(sys.argv[1]) as fichierOptions:
    options = json.load(fichierOptions)

print("Récupération des images... \n")

l_im = options['tailleImageY']
h_im = options['tailleImageX']
df, df_proj, pca = ouvrir(options['DossierRecupImages'], h_im, l_im, options["Nombre_axe_ACP"])

print("##### Récupération des images réussie ##### \n")

print("Traitement des images par la méthode :",  type(options["Methode"]), "...  \n")

nom_dossier = options['DossierNouvellesImages'] + '/' + "results_" + heure

if not os.path.exists(nom_dossier):
    os.makedirs(nom_dossier)

df_score = pd.DataFrame()

with alive_bar(options['NbImagesACreer']) as bar:
    if (options['Methode'] == 6):
        X = np.random.choice(df.shape[0],4,replace = False)
        for i in range(5):
            for j in range(5):
                im = tableau_multi_images(df_proj, i, j, X, pca)
                save(str(i) + "_" + str(j), nom_dossier, im, options["Afficher_images"])
                df_score["score-im_" + str(i) + "_" + str(j) ] = compare(im.reshape(h_im, l_im), df)
                bar()
    else :
        for i in range (options['NbImagesACreer']):
            im = traitement(df_proj, pca, l_im, h_im, options["Nombre_axe_ACP"], options['Methode'])
            save(str(1) + "_" + str(i), nom_dossier, im, options["Afficher_images"])
            df_score["score-im_" + str(i)] = compare(im.reshape(h_im, l_im), df)
            bar()

df_score.to_csv(nom_dossier + "/score_" + heure + '.csv')

if (options["Affichage_score"]) :
    print("\n")
    if (options['Methode'] != 6) :
        for i in range (options['NbImagesACreer']) : 

            print("Image n°", str(i), " :")
            print("     - score moyen. : ", df_score.iloc[:,i].mean())
            print("     - score min. : ", df_score.iloc[:,i].min())
            print("     - score max. : ", df_score.iloc[:,i].max())
    
    elif (options['Methode'] == 6) :
        for i in range (25) : 

            print("Image n°", str(i), " :")
            print("     - score moyen : ", df_score.iloc[:,i].mean())
            print("     - score minimal : ", df_score.iloc[:,i].min())
            print("     - score maximal : ", df_score.iloc[:,i].max())

    print("\n Scores des images : \n")
    print("   - Score moyen :", df_score.mean().mean())
    print("   - Score maximal :", df_score.max().max())
    print("   - Score minimal :", df_score.min().min())


print("\n ##### Traitement des images réussie #####  \n")
print("Les images créées sont dans le dossier : ", nom_dossier, '\n')

print("Temps de compilation : ", time.time() - start_time)
