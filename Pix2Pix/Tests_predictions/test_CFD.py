

"""
script qui test la performance du meilleur modèle entrainé 

stratégie d'entrainement :

    - dataset train = 720 couples d'images (input, taget) du film 3 mélangées random
    - dataset val = 180 couples d'images (input, taget) du film 3 mélangées random
    - dataset test = 90 couples d'images (input, taget) du film 3 mélangées random

Workflow :

1- récupération du modèle ayant la loss la plus faible
2- loading G, D et best_criteria à partir du checkpoint enregistré par Ray
3- génération des prédictions du dataset de test
4- calcule de la MAE sur les vitesses sur les predictions individuelles
5- calcule des MAE globales sur les datasets de train/val/test
6- affichage de la courbe des erreurs
7- affichage des predictions et du mapping des erreurs


"""




from pathlib import Path
import os
import numpy as np
from CFD_dataset_cas_2 import load_dataset, get_tf_dataset, resize_and_normalize
import matplotlib.pyplot as plt
import json
import pandas as pd
import tensorflow as tf
import keras




DEBUG = False

# 1- Datasets

dataset_name = "film_3.npy"
dataset_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/") / dataset_name
dataset_path = dataset_path.resolve()

inputs, targets, train_idx, test_idx, val_idx = load_dataset(dataset_path, offset=1)
x_train, y_train = resize_and_normalize(inputs[train_idx], targets[train_idx])
x_val, y_val = resize_and_normalize(inputs[val_idx], targets[val_idx])
x_test, y_test = resize_and_normalize(inputs[test_idx], targets[test_idx])
test_dataset = get_tf_dataset(x_test, y_test, BATCH_SIZE=1, shuffle=False)
train_dataset = get_tf_dataset(x_train, y_train, BATCH_SIZE=1, shuffle=False)
val_dataset = get_tf_dataset(x_val, y_val, BATCH_SIZE=1, shuffle=False)

if DEBUG:
    print("shape :", x_train.shape, y_train.shape)
    print("min :", np.min(x_train), np.min(y_train))
    print("max :", np.max(x_train), np.max(y_train))
    print("type :", type(x_train[0]), type(y_train[0]))
    


# 2- variables des bornes du film

CFD_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/")
with open(os.path.join(CFD_path, "bornes_film_3.json"),"r") as f:
    bornes = json.load(f)

v_min = bornes["v_min"]
v_max = bornes["v_max"]





# 3- identification du meilleur checkpoint

ray_results_dir = os.path.abspath(
    "./my_ray_results_cas_8/cas_8_run_1")
os.makedirs(ray_results_dir, exist_ok=True)
print(ray_results_dir)

dataframes = []

for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        results_file = os.path.join(trial_path, "result.json")
        params_file = os.path.join(trial_path, "params.json")
        if os.path.isfile(params_file) and os.path.isfile(results_file):
            # Vérifie si les fichiers sont vides
            if os.path.getsize(params_file) == 0:
                print(f"Fichier vide ignoré : {params_file}")
                continue
            if os.path.getsize(results_file) == 0:
                print(f"Fichier vide ignoré : {results_file}")
                continue
            try:
                with open(params_file, "r") as f:
                    params = json.load(f)
                # Lecture du fichier résultats directement via le chemin
                result_json = pd.read_json(results_file, lines=True)
                result_json["trial_id"] = trial_dir
                dataframes.append(result_json)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                print(f"Erreur de lecture dans {trial_dir} : {e}")
                continue


if not dataframes:
    raise ValueError("Aucun fichier de résultats valide trouvé.")
# je réunis les trials dans un dataframe
result_df = pd.concat(dataframes, ignore_index=True)
# je recupère uniquement la métrique la plus performante / trial
df_best = result_df.loc[result_df.groupby(
    "trial_id")["best_criteria"].idxmin()]
print(df_best.shape[0])
# je trie par ordre ascendant
df_best_sorted = df_best.sort_values("best_criteria")
# Identifie le meilleur entraînement
best_trial = df_best_sorted.iloc[0]
print("====== Meilleur entraînement ======")
print(f"Trial ID        : {best_trial['trial_id']}")
print(f"Best criteria   : {best_trial['best_criteria']}")
print(f"Training step   : {best_trial['training_iteration']}")
best_trial_id = best_trial['trial_id']
params_path = os.path.join(ray_results_dir, best_trial_id, "params.json")
if os.path.isfile(params_path):
    with open(params_path, "r") as f:
        best_params = json.load(f)
    print("------ Hyperparamètres ------")
    for k, v in best_params.items():
        print(f"{k}: {v}")
# checkpoint path
best_trial_path = os.path.join(ray_results_dir, best_trial_id)
checkpoint_dirs = [
    d for d in os.listdir(best_trial_path)
    if os.path.isdir(os.path.join(best_trial_path, d)) and d.startswith("checkpoint_")
]
if not checkpoint_dirs:
    raise FileNotFoundError("Aucun dossier de checkpoint trouvé.")
# Option 1 : prendre le dernier checkpoint (ordre alphanumérique)
checkpoint_dirs.sort()
latest_checkpoint = checkpoint_dirs[-1]
# Chemin complet du dernier checkpoint
checkpoint_path = os.path.join(best_trial_path, latest_checkpoint)
print("Chemin du checkpoint :", checkpoint_path)




# 4- loading du meilleur modèle entrainé

generator = keras.models.load_model(os.path.join(
    checkpoint_path, "generator.keras"))
discrimintor = keras.models.load_model(os.path.join(
    checkpoint_path, "discriminator.keras"))
best_criteria_file = os.path.join(
    checkpoint_path, "training_state.json")



# 4- évaluation du modèle

output_dir = os.path.abspath("./predictions_pix2pix_CFD_cas_8")
os.makedirs(output_dir, exist_ok=True)

mae_loss = tf.keras.losses.MeanAbsoluteError()

def prediction(initial_image, target_image, v_min, v_max):
    """
    _Summary_: fonction qui génère une prediction et calcule l'erreur sur la vitesse de prediction vs vitesse cible
    _Args_: 
        - initial_image (np.array): image d'entrée (shape (1,H,W,1))
        - target_image (np.array): image cible (shape (1,H,W,1))
        - v_min, v_max (float): bornes de la vitesse physique
    _Returns_: 
        - prediction (np.array) : image predite [0,255]
        - MAE de la vitesse (float)
    
    """
    current_pred = generator(initial_image, training=False)  # tenseur (b, C, W, H)
    pred = ((current_pred[0].numpy()+1) * 127.5) # np.darray (H,W,1), [0,255] /// on dénormalise
    pred_vitesse = pred/255*(v_max-v_min)+v_min
    tar_vitesse = (target_image[0].numpy()+1)/2*(v_max-v_min)+v_min
    mae_vitesse = np.mean(np.abs(pred_vitesse - tar_vitesse))
    return pred, mae_vitesse


if DEBUG:
    inp = x_test[0:1] 
    tar = y_test[0:1]
    print(inp.shape, tar.shape, type(inp), type(tar), np.min(inp), np.max(inp)) # tensors ([1,128,128,1]) [-1,1]
    tar_arr = ((tar.numpy()+1)*127.5) # np.ndarray (1,128,128,1) [0,255]
    tar_arr_3d = ((tar[0].numpy()+1)*127.5) # (128,128,1) [0,255] on supp la dimension batch tar[0]
    pred, loss_mae = prediction(inp, tar, v_min, v_max)
    print(pred.shape, type(pred), np.min(pred), np.max(pred)) # np.array (128,128,1) [0,255]
    print("MAE_loss :", loss_mae)
    # pour visualiser les images on ramène en 2D et on dénormalise
    image_inp = inp[0,:,:,0] # tensor ([128,128]) [-1,1]
    img_inp = ((image_inp.numpy()+1)*127.5) # numpy.ndarray [0,255]
    img_inp = img_inp.astype(np.uint8)
    plt.imshow(img_inp)
    plt.show()   
    # pour visualiser la prediction on doit supprimer la dernière = (H,W)
    image = pred[:,:,0].astype(np.uint8) # ou image = pred.squeeze().astype(np.uint8)
    plt.imshow(image)
    plt.show()

# on va calculer l'erreur individuelle sur chacune des predictions du dataset de test
mae_list = []
for i in range(x_test.shape[0]):
    inp = x_test[i:i+1] # tensor ([1,128,128,1]) normalise
    tar = y_test[i:i+1]
    pred, loss_mae = prediction(inp, tar, v_min, v_max)
    mae_list.append(loss_mae)


# prediction adaptée à un dataset

def prediction_dataset(inputs, targets, v_min, v_max):
    """
    
    """
    mae_list = []
    preds_list = []
    for i in range(inputs.shape[0]):
        inp = inputs[i:i+1] # tensor ([1,128,128,1]) normalise
        tar = targets[i:i+1]
        pred, loss_mae = prediction(inp, tar, v_min, v_max)
        mae_list.append(loss_mae)
        preds_list.append(pred[np.newaxis, ...]) # oblige à garder la dimension batch [N,128,128,1]
    mae_dataset = np.mean(mae_list)
    preds = np.concatenate(preds_list, axis=0)
    return preds, mae_dataset



# on veut calculer la valeur de la mae sur chacun des datasets
preds_test, mae_dataset_test = prediction_dataset(x_test, y_test, v_min, v_max)
preds_train, mae_dataset_train = prediction_dataset(x_train, y_train, v_min, v_max)
preds_val, mae_dataset_val = prediction_dataset(x_val, y_val, v_min, v_max)

if DEBUG:
    print("MAE Test :", mae_dataset_test)
    print("MAE Train :", mae_dataset_train)
    print("MAE Val :", mae_dataset_val)
    print("Shape preds_test :", preds_test.shape)

# courbe des pertes
plt.figure(figsize=(10, 6))
plt.plot(mae_list, marker='o')
plt.xlabel("Index image")
plt.ylabel("Loss (MAE)")
plt.title("Courbe des pertes sur le dataset de test")
plt.grid(True)
plt.axhline(mae_dataset_test,color='red', linestyle='--', linewidth=2, label=f"perte_test: {mae_dataset_test:.4f}")
plt.axhline(mae_dataset_train,color='green', linestyle='--', linewidth=2, label=f"perte_train: {mae_dataset_train:.4f}")
plt.axhline(mae_dataset_val,color='blue', linestyle='--', linewidth=2, label=f"perte_val: {mae_dataset_val:.4f}")
plt.legend(loc='lower left', bbox_to_anchor=(0.2, 0.8), fontsize=9)
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_test.png"))
plt.tight_layout()
plt.show()



# affichage des images


for i in range (0, x_test.shape[0], 50):
    # Données d'entrée et cible
    input_i = x_test[i]
    tar_i = y_test[i]
    pred_i = preds_test[i]
    # Conversion en images [0,255]
    img_inp = np.squeeze((input_i.numpy() + 1) * 127.5).astype(np.uint8)  # (128,128)
    img_tar = np.squeeze((tar_i.numpy() + 1) * 127.5).astype(np.uint8)
    img_pred = np.squeeze(pred_i).astype(np.uint8)
    # Conversion en vitesses physiques
    pred_vitesse = np.squeeze(pred_i) / 255 * (v_max - v_min) + v_min
    tar_vitesse = np.squeeze((tar_i.numpy() + 1) / 2) * (v_max - v_min) + v_min
   # Carte d’erreur
    error_map = np.abs(pred_vitesse - tar_vitesse)
    # affichage
    fig, ax = plt.subplots (1,4,sharey=True,figsize=(10, 4))
    ax[0].imshow(img_inp)
    ax[0].set_title(f'Input {i}')
    ax[1].imshow(img_tar)
    ax[1].set_title(f'Target {i}')
    ax[2].imshow(img_pred)
    ax[2].set_title(f'Prediction {i}')
    im=ax[3].imshow(error_map, cmap="hot")
    ax[3].set_title(f'Erreur_prediction_vitesses')
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    fig.suptitle(f'Prediction Example {i}', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_{i}.png'))
    plt.show()
    plt.close()























