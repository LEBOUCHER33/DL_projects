"""

Script de preprocessing des data pour évaluer les performances du modèle entrainé sur les champs de vitesse

= comparer les vitesses interpolées en chaque points de la grille approximant les valeurs réelles aux noeuds de maillage

1- on a un maillage irrégulier avec des coordonnées (x_i,y_i) 
2- en chaque pt ce de maillage on a des vecteurs de vitesse (vx_i, vy_i)
3- on crée une grille régulière 2D
4- en chaque point de cette grille d'interpolation on va évaluer la vitesse du pt de maillage associé :
    - on crée une triangulation pour relier les pts de maillages à la grille
    - à chacun des pts on évalue la norme de vitesse


=> la valeur de chaque pixel = norme du vecteur de vitesse interpolées en un pt
=> les images sont normalisées [0,255]
=> les prédictions de Pix2Pix sont les normes des vecteurs de vitesse 

Objectif :

1- création d'un film d'images interpolées à partir du maillage
2- calculer et sauvegarder les valeurs min et max de la grille d'interpolation
2- sauvegarde du film et des bornes min et max de la grille d'interpolation

"""

import os
from pathlib import Path
import numpy as np
from matplotlib.tri import Triangulation, LinearTriInterpolator
import json


# /////////////////////////////////
# Création d'un film d'images interpolées
# /////////////////////////////////

# 1- on crée un dataset = 1 film d'images interpolées à partir des normes des vitesses

dataset_path = os.path.abspath('./Datasets/Eagle_dataset/Cre/3/1')
output_dir = "./git/ImageMLProject/Datasets/CFD_Dataset"
os.makedirs(output_dir, exist_ok=True)


# fonction de loading des data du dataset

def load_from_npz(path: Path) -> np.ndarray:
    """
    _Summary_ : loading des fichiers numpy() des données de simulation

    _Args_ : path du dataset

    _Returns_: 
                    - coordonnées des points de maillage (np.darray) 
                    - indices des sommets des triangles du maillage (np.darray)
                    - type de chaque noeud (0 à 6) (np.darray)
                    - vecteurs vitesses (vx, vy) de chaque noeud (np.darray)
                    - pressions pour chaque noeud

    """
    data = np.load(os.path.join(path, "sim.npz"))
    cells = np.load(os.path.join(path, "triangles.npy"))
    mesh_pos = data["pointcloud"][0: 990].copy()
    cells = cells[0: 990]
    Vx = data["VX"][0:990].copy()
    Vy = data["VY"][0:990].copy()
    Ps = data["PS"][0:990].copy()
    Pg = data["PG"][0:990].copy()
    node_type = data["mask"][0:990].copy()
    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    return mesh_pos, cells, node_type, velocity, pressure



# fonction de processing des data

def get_list_image(path : Path, resolution=512, T=990) -> np.ndarray :
    """
_Summary_: fonction qui prend un chemin vers un dossier et renvoie la liste des images du film de simulation
		1- récupère les données des fichiers .npy et .npz du film
		2- pour chacun des noeuds de maillages : effectue une triangulation (maillage irrégulier non-structuré)
		3- fait une interpolation de la vitesse sur une grille carré régulière pixelaire 500x500
		4- normalise les valeurs de la grille 
        5- définit les valeurs min et max de la grille


_Args_: path (Path) : path vers le dossier contenant lgrids.e film de simulation (990 images):	
				- 1 fichier .npz
				- 1 fichier .npy

_Returns_: 
        - 990 matrices/images de taille 500x500 (np.ndarray, (990,500,500))
        - les valeurs min et max de la grille


    """ 
    data = load_from_npz(path)
    m, c, n, v, p = data
# calcul de la vitesse : calcul de la norme de chaque vecteur de vitesse
    velo = (v**2).sum(-1)
# sélection des points de maillage : on récupère les coordonnées des pts de maillage à t=0
    x = m[0, :, 0]
    y = m[0, :, 1]
    triangles = c[0]
# Filtrage des triangles invalides
    valid_triangles = triangles[np.all(triangles < len(x), axis=1)] # élimine les indices hors limites (>3464)
    valid_triangles = np.array([tri for tri in valid_triangles if len(set(tri)) == 3])  # anti-triangles dégénérés    
# Triangulation : construction d'une structure de triangulation
    tri = Triangulation(x, y, valid_triangles)
# Interpolateur
    interp_lin = LinearTriInterpolator(tri, velo[0])
# Définir la grille carrée (résolution) -- Centres les plus à gauche, à droite, en haut en bas
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
# Création d'une grille carrée (les pixels en gros)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, num=resolution),
        np.linspace(y_min, y_max, num=resolution))
# Interpolation sur la grille
    grid_z = interp_lin(grid_x, grid_y)
    grids = []
    velo = (v**2).sum(-1)
    for t in range(T):
        interpolator = LinearTriInterpolator(tri, velo[t])
        grid_z = interpolator(grid_x, grid_y)
        grid_z = np.nan_to_num(grid_z, nan=-velo.max())  # remplacer les NaN par une valeur négative
        grids.append(grid_z.data)
    grids = np.array(grids)
    v_min, v_max = grids.min(), grids.max()
    grids_norm = (grids-grids.min())/(grids.max()-grids.min())*255 # normalisation globale
    grids_norm = grids_norm.astype("int")
    return grids, grids_norm, v_min, v_max


film_3_phy, film_3_norm, v_min, v_max = get_list_image(dataset_path, resolution=128)

# sauvegarde du film
np.save(os.path.join(output_dir, "film_3.npy"), film_3_norm)
np.save(os.path.join(output_dir, "film_3_phy.npy"), film_3_phy)


# sauvegarde des valeurs des bornes d'interpolation
bornes = {"v_min": float(v_min), "v_max": float(v_max)}
with open(os.path.join(output_dir, "bornes_film_3.json"), "w") as f:
        json.dump(bornes, f, indent=4)


