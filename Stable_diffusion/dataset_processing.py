"""
Script de processing du dataset de simulation CFD pour fine_tuning de Stable_diffusion
- dataset : film de simulation de CFD

- processing du fichier .npy en images RGB 512x512: 
    - load du fichier .npy
    - création des paires input/target avec un offset de 1
    - normalisation en [0,255]
    - split en 3 datasets (train/val/test) avec un découpage 60/20/20
    - sauvegarde des images + caption (=text) dans des dossiers structurés pour la lib datasets

"""


import numpy as np
from pathlib import Path
import os
from PIL import Image


# 1- définir le dataset

dataset_name = "images_normalised_1_1.npy"
dataset_path = Path(
    "./git/ImageMLProject/Datasets/") / dataset_name

dataset_path = dataset_path.resolve()



# 2- processing du dataset

# création du dossier des data
data_dir = "./git/ImageMLProject/Stable_diffusion/data"
os.makedirs(data_dir, exist_ok=True)

# processing des images
def process_dataset(path: Path, data_dir, offset=1, split=(0.6, 0.2, 0.2), seed=42):
    """
    _Summary_: process un dataset au format sp
    
    - récupère les grilles interpolées d'un film de simulation depuis un fichier .npy 
    - crée trois datasets (train/val/test) avec un décalage entre input et target et un découpage 60/20/20 par défaut

    _Args_: 
	- path (Path) : path du fichier .npy 
	- offset (int): décalage entre les inputs et les targets (par défaut offset de 1).
	- split (tuple): proportion (train, val, test)
	- seed (int): graine aléatoire pour reproductibilité (optionnel)


    _Returns_: None

    """
    np.random.seed(seed)
    data = np.load(path) 
    print("Dataset originel : ", data.shape) #(990,500,500)
    # création des paires et ajout d'une dimension
    inputs = data[:-offset]  # (N - offset, H, W)
    targets = data[offset:]  # (N - offset, H, W)
    # normalisation des images en [0,255]
    inputs = ((inputs - inputs.min()) / (inputs.max() - inputs.min()) * 255).astype(np.uint8)
    targets = ((targets - targets.min()) / (targets.max() - targets.min()) * 255).astype(np.uint8)
    # split des images dans les 3 datasets
    total = inputs.shape[0]
    train_end = int(split[0] * total)
    val_end   = train_end + int(split[1] * total)
    splits = {
        "train": (inputs[:train_end],  targets[:train_end]),
        "val":   (inputs[train_end:val_end], targets[train_end:val_end]),
        "test":  (inputs[val_end:],   targets[val_end:]),
    }
    for split_name, (x, y) in splits.items():
        split_dir = os.path.join(data_dir, split_name, "images")
        os.makedirs(split_dir, exist_ok=True)
        for i in range(len(x)):
            # Convertir en image RGB 512x512
            inp = Image.fromarray(x[i]).convert("RGB").resize((512, 512))
            tgt = Image.fromarray(y[i]).convert("RGB").resize((512, 512))
            # Sauvegarder : input, target, + caption
            inp.save(f"{split_dir}/{i:05d}_input.png")
            tgt.save(f"{split_dir}/{i:05d}_target.png")
            with open(f"{split_dir}/{i:05d}.txt", "w") as f:
                f.write("Simulation fluide CFD, prédiction dynamique")



