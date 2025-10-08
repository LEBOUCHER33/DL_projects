


"""

Script qui définit les fonctions qui permettent de loader et preprocesser 70% du dataset de CFD au hasard

Dataset de train : 720 images  // sert à entrainer le modèle
Dataset de validation : 180 images // sert à évaluer les performances pendant l'entrainement
Dataset de test : 89 images   // sert à l'évaluation finale du modèle

1- loading du fichier .npy
2- création des paires d'images (entrée/cible)
3- répartition aléatoire des paires dans les 3 datasets


"""



import tensorflow as tf
import numpy as np
from pathlib import Path


# //////////////////////////////
# loading et processing du dataset
# //////////////////////////////




def load_dataset(path: Path, offset=1, train_size=720, val_size=180, seed=42):
    """
    _Summary_: récupère les grilles interpolées d'un film de simulation depuis un fichier .npy et crée trois datasets (train/val/test) avec un décalage entre input et target

    _Args_: 
	- path (Path) : path du fichier .npy 
	- offset (int): décalage entre les inputs et les targets (ex: 5, 10, 20...).
	- train_size (int): taille du dataset d'entraînement.
        - val_size (int): taille du dataset de validation.


    _Returns_: train_ds, test_ds, val_ds : tuples de tf.Tensor (inputs, targets)

    """
    np.random.seed(seed)
    data = np.load(path)
    print(data.shape)
    #création des paires et ajout d'une dimension
    inputs = data[:-offset]  # (N - offset, H, W)
    targets = data[offset:]  # (N - offset, H, W)
    inputs = inputs[..., np.newaxis]
    targets = targets[..., np.newaxis]  # (N - offset, H, W, 1) // (batch_size, height, width, channels) // lot d'images en 2D et niveaux de gris = format attendu pour Pix2Pix
    # Mélange les indices de façon aléatoire
    total = inputs.shape[0]  # = N - offset
    indices = np.random.permutation(total)  # mélange les données aléatoirement avant de séparer dans les datasets // génère un tableau d'indices mélangés
    train_idx = indices[:train_size]  # récupère train_size images
    val_idx = indices[train_size:train_size + val_size]  # récupère val_size images
    test_idx = indices[train_size + val_size:]   # récupère (train_size - val_size) images
    return inputs, targets, train_idx, test_idx, val_idx


def resize_and_normalize(x, y, image_size = (128,128)):
        x = tf.image.resize(x, image_size)
        y = tf.image.resize(y, image_size)
        x = (x / 127.5) - 1.0  # Normalisation [-1, 1]
        y = (y / 127.5) - 1.0
        return x, y


def get_tf_dataset(inputs, targets, BATCH_SIZE=1, shuffle=True):
    """
    _Summary_: Récupère les images formatées du dataset train,les mélange et les répartit dans les lots pour l'entrainement du modèle

    _Args_: 
        - le path du dataset (fichier .npy), 
        - le nombre d'images par lot (int), 
        - si on veut appliquer la fonction shuffle de tf (bool)
        - le nombre d'images du dataset (int)

    _Returns_: le dataset d'images formatées, mélangées, organisées par lot (PrefetchDataset)

    """
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(inputs))
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset











