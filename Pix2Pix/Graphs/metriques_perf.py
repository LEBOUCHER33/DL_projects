from pathlib import Path
import os
import numpy as np
from Pix2Pix.Datasets_Processing.dataset_CFD import load_dataset, get_tf_dataset, resize_and_normalize
import json
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


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


DEBUG = False

# 1- Datasets

dataset_name = "film_3_128x128.npy"
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
with open(os.path.join(CFD_path, "data_film_3_128x128.json"),"r") as f:
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
discriminator = keras.models.load_model(os.path.join(
    checkpoint_path, "discriminator.keras"))
best_criteria_file = os.path.join(
    checkpoint_path, "training_state.json")



# 4- évaluation du modèle

output_dir = os.path.abspath("./predictions_pix2pix_CFD_cas_8")
os.makedirs(output_dir, exist_ok=True)



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
    # prediction
    current_pred = generator(initial_image, training=False)  # tenseur (b, C, W, H) [-1,1]
    pred = ((current_pred[0].numpy()+1) * 127.5) # np.darray (H,W,1), [0,255] /// on dénormalise
    # conversion en vitesse physique
    pred_vitesse = pred/255*(v_max-v_min)+v_min # [0,255] -> [0,1] -> [v_min,v_max]
    tar_vitesse = (target_image[0].numpy()+1)/2*(v_max-v_min)+v_min # [-1,1] -> [0,1] -> [v_min,v_max]
    # calculs d'erreurs
    diff_vitesse = pred_vitesse - tar_vitesse
    err_mae = np.abs(diff_vitesse) # np.ndarray (H,W,1) contenant l'erreur absolue pixel par pixel
    err_mse = np.square(diff_vitesse)
    # moyennes des erreurs et dispersion
    mae_mean = np.mean(err_mae) # erreur moyenne sur l'image
    mse_mean = np.mean(err_mse)
    rmse_mean = np.sqrt(mse_mean)
    var_mae = np.var(err_mae)
    return pred, mae_mean, mse_mean, rmse_mean, var_mae




if DEBUG:
    inp = x_test[0:1] 
    tar = y_test[0:1]
    print(inp.shape, tar.shape, type(inp), type(tar), np.min(inp), np.max(inp)) # tensors ([1,128,128,1]) [-1,1]
    tar_arr = ((tar.numpy()+1)*127.5) # np.ndarray (1,128,128,1) [0,255]
    tar_arr_3d = ((tar[0].numpy()+1)*127.5) # (128,128,1) [0,255] on supp la dimension batch tar[0]
    pred, loss_mae, loss_mse, loss_rmse, var_mae = prediction(inp, tar, v_min, v_max)
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



# on va calculer l'erreur individuelle sur chacune des predictions du dataset de test et la variance
error_maps = []
mae_list = []
rmse_list = []
for i in range(x_test.shape[0]):
    inp = x_test[i:i+1] # tensor ([1,128,128,1]) normalise
    tar = y_test[i:i+1]
    pred, test_mae, test_mse, test_rmse, var_mae = prediction(inp, tar, v_min, v_max)
    mae_list.append(test_mae)
    rmse_list.append(test_rmse)
    





# prediction adaptée à un dataset

def prediction_dataset(inputs, targets, v_min, v_max):
    """
    
    """
    mae_list = []
    mse_list = []
    rmse_list = []
    var_list = []
    preds_list = []
    for i in range(inputs.shape[0]):
        inp = inputs[i:i+1] # tensor ([1,128,128,1]) normalise
        tar = targets[i:i+1]
        pred, mae_mean, mse_mean, rmse, var_mae = prediction(inp, tar, v_min, v_max)
        mae_list.append(mae_mean)
        mse_list.append(mse_mean)
        rmse_list.append(rmse)
        var_list.append(var_mae)
        preds_list.append(pred[np.newaxis, ...]) # oblige à garder la dimension batch [N,128,128,1]
    mae_dataset = np.mean(mae_list) # moyenne des MAE individuelles sur le dataset
    mse_dataset = np.mean(mse_list)
    rmse_dataset = np.sqrt(mse_dataset)
    var_dataset = np.var(mae_list) # variance des MAE individuelles sur le dataset
    preds = np.concatenate(preds_list, axis=0) 
    var_mae = np.var(mae_list) # variance des MAE individuelles sur le dataset
    return preds, mae_dataset, rmse_dataset, mae_list, rmse_list, var_dataset



# on veut calculer la valeur de la mae sur chacun des datasets
preds_test, mae_dataset_test, rmse_dataset_test, test_mae_list, test_rmse_list, test_var = prediction_dataset(x_test, y_test, v_min, v_max)
preds_train, mae_dataset_train, rmse_dataset_train, train_mae_list, train_rmse_list, train_var = prediction_dataset(x_train, y_train, v_min, v_max)
preds_val, mae_dataset_val, rmse_dataset_val, val_mae_list, val_rmse_list, val_var = prediction_dataset(x_val, y_val, v_min, v_max)

if DEBUG:
    print("MAE Test :", mae_dataset_test)
    print("MAE Train :", mae_dataset_train)
    print("MAE Val :", mae_dataset_val)
    print("Shape preds_test :", preds_test.shape)

# //////////////////////
# courbe des pertes
# //////////////////////
def plot_loss_curve(metric_name, values_list, dataset_means, output_dir):
    """
    Affiche et sauvegarde la courbe des pertes (MAE, MSE, RMSE) sur le dataset de test.

    Args:
        metric_name (str): Nom de la métrique ("MAE", "MSE" ou "RMSE").
        values_list (list): Liste des pertes individuelles par image.
        dataset_means (dict): Moyennes globales des pertes {"train": x, "val": y, "test": z}.
        output_dir (str): Dossier de sauvegarde.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(values_list, marker='o', markersize=3, linewidth=1, color='tab:blue', label=f'{metric_name} par image')

    plt.xlabel("Index image")
    plt.ylabel(f"Loss ({metric_name})")
    plt.title(f"Courbe des pertes {metric_name} sur le dataset de test")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Lignes horizontales pour les moyennes globales
    plt.axhline(dataset_means['test'], color='red', linestyle='--', linewidth=2,
                label=f"{metric_name}_test: {dataset_means['test']:.4f}")
    plt.axhline(dataset_means['train'], color='green', linestyle='--', linewidth=2,
                label=f"{metric_name}_train: {dataset_means['train']:.4f}")
    plt.axhline(dataset_means['val'], color='blue', linestyle='--', linewidth=2,
                label=f"{metric_name}_val: {dataset_means['val']:.4f}")

    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout()

    filename = f"courbe_des_pertes_{metric_name.lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# dict des métriques d'erreurs à afficher
metrics = {
    "MAE": {
        "values": mae_list,
        "means": {
            "train": mae_dataset_train,
            "val": mae_dataset_val,
            "test": mae_dataset_test
        }
    },
    "RMSE": {
        "values": rmse_list,
        "means": {
            "train": rmse_dataset_train,
            "val": rmse_dataset_val,
            "test": rmse_dataset_test
        }
    }
}

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
for ax, (name, data) in zip(axes, metrics.items()):
    ax.plot(data["values"], marker='o', markersize=3, color='tab:blue', label=name)
    for split, color in zip(["train", "val", "test"], ["green", "blue", "red"]):
        ax.axhline(data["means"][split], color=color, linestyle='--', label=f"{split}: {data['means'][split]:.4f}")
    ax.set_ylabel(f"{name}")
    ax.grid(True, linestyle='--', alpha=0.5)
axes[-1].set_xlabel("Index image")
fig.suptitle("Courbes de pertes par métrique")
fig.tight_layout()
fig.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_metriques.png"), dpi=300)
plt.show()





# ////////////////////////////////////////////
# courbe des pertes globales sur les trois datasets avec affichage de la variance globale
# ////////////////////////////////////////////


all_data = []

# on regroupe les données
for idx, mae in zip (train_idx, train_mae_list):
    all_data.append((idx, mae, 'train'))

for idx, mae in zip (val_idx, val_mae_list):
    all_data.append((idx, mae, 'val'))

for idx, mae in zip (test_idx, test_mae_list):
    all_data.append((idx, mae, 'test'))    

# on trie selon l'index d'origine
all_data.sort(key=lambda x: x[0])

# tableaux numpy
indices = np.array([item[0] for item in all_data])
mae_values = np.array([item[1] for item in all_data])
labels = np.array([item[2] for item in all_data])   


# figure
plt.figure(figsize=(12, 6))
colors = {'train': 'green', 'val': 'blue', 'test': 'red'}
for split in ['train', 'val', 'test']:
    mask = labels == split
    plt.errorbar(indices[mask], 
                 mae_values[mask], 
                 fmt='o', markersize=4, 
                 label=split, 
                 color=colors[split], alpha=0.7)

plt.axhline(mae_dataset_train, color='green', linestyle='--', label=f"MAE_train: {mae_dataset_train:.4f} m/s")
plt.fill_between(indices, mae_dataset_train - train_var, mae_dataset_train + train_var, color='green', alpha=0.2, label=f"Variance_train : {train_var:.4f} (m/s)²")
plt.axhline(mae_dataset_val, color='blue', linestyle='--', label=f"MAE_val: {mae_dataset_val:.4f} m/s")
plt.fill_between(indices, mae_dataset_val - val_var, mae_dataset_val + val_var, color='blue', alpha=0.2, label=f"Variance_val : {val_var:.4f} (m/s)²")
plt.axhline(mae_dataset_test, color='red', linestyle='--', label=f"MAE_test: {mae_dataset_test:.4f} m/s")
plt.fill_between(indices, mae_dataset_test - test_var, mae_dataset_test + test_var, color='red', alpha=0.2, label=f"Variance_test : {test_var:.4f} (m/s)²")
plt.xlabel("Index image dans le film")
plt.ylabel("MAE par image (moyenne des erreurs absolues sur la vitesse) m/s")
plt.title("Courbe des pertes MAE moyenne par prediction sur les trois datasets")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
#plt.savefig(os.path.join(output_dir, "courbe_des_pertes_mae_avec_variance.png"), dpi=300)
plt.show()    


# //////////////////////////////////
# courbe des RMSE moyens sur les trois datasets
# //////////////////////////////////

all_data_rmse = []

# on regroupe les données
for idx, rmse in zip (train_idx, train_rmse_list):
    all_data_rmse.append((idx, rmse, 'train'))

for idx, rmse in zip (val_idx, val_rmse_list):
    all_data_rmse.append((idx, rmse, 'val'))

for idx, rmse in zip (test_idx, test_rmse_list):
    all_data_rmse.append((idx, rmse, 'test'))    

# on trie selon l'index d'origine
all_data_rmse.sort(key=lambda x: x[0])

# tableaux numpy
indices = np.array([item[0] for item in all_data_rmse])
rmse_values = np.array([item[1] for item in all_data_rmse])
labels = np.array([item[2] for item in all_data_rmse])   

# figure
plt.figure(figsize=(12, 6))
colors = {'train': 'green', 'val': 'blue', 'test': 'red'}
for split in ['train', 'val', 'test']:
    mask = labels == split
    plt.errorbar(indices[mask], 
                 rmse_values[mask], 
                 fmt='o', markersize=4, 
                 label=split, 
                 color=colors[split], alpha=0.7)

plt.axhline(rmse_dataset_train, color='green', linestyle='--', label=f"RMSE_train: {rmse_dataset_train:.4f} m/s")
plt.axhline(rmse_dataset_val, color='blue', linestyle='--', label=f"RMSE_val: {rmse_dataset_val:.4f} m/s")
plt.axhline(rmse_dataset_test, color='red', linestyle='--', label=f"RMSE_test: {rmse_dataset_test:.4f} m/s")
plt.xlabel("Index image dans le film")
plt.ylabel("RMSE par image (moyenne des erreurs quadratiques sur la vitesse) m/s")
plt.title("Courbe des pertes RMSE moyenne par prediction sur les trois datasets")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_rmse.png"), dpi=300)
plt.show()    






# ////////////////////////////////////////////////
# affichage des images / des erreurs absolues / des erreurs quadratiques
# ///////////////////////////////////////////////


for i in range (3, x_test.shape[0], 30):
    # Données d'entrée et cible
    input_i = x_test[i]
    tar_i = y_test[i]
    pred_i = preds_test[i]
    # Conversion en images [0,255]
    img_inp = np.squeeze((input_i.numpy() + 1) * 127.5).astype(np.uint8)  # (128,128)
    img_tar = np.squeeze((tar_i.numpy() + 1) * 127.5).astype(np.uint8)
    img_pred = np.squeeze(pred_i).astype(np.uint8)
    # Conversion en vitesses physiques
    pred_vitesse = np.squeeze(pred_i) / 255 * (v_max - v_min) + v_min # [0,255] -> [0,1] -> [v_min,v_max]
    tar_vitesse = np.squeeze((tar_i.numpy() + 1) / 2) * (v_max - v_min) + v_min # [-1,1] -> [0,1] -> [v_min,v_max]
   # Carte d’erreur mae : erreur locale absolue
    error_map_mae_rel = np.abs(pred_vitesse - tar_vitesse) / (np.maximum(0, np.abs(tar_vitesse)))
    error_map_mae_abs = np.abs(pred_vitesse - tar_vitesse)
    error_max = np.max(error_map_mae_abs)
    error_min = np.min(error_map_mae_abs)
    error_variance = np.var(error_map_mae_abs)
    # Carte d’erreur rmse : erreur locale quadratique
    error_map_quad = np.square(pred_vitesse - tar_vitesse)
    error_map_quad_root = np.sqrt(error_map_quad)
    # statistiques globales sur l'image
    mae_mean = np.mean(error_map_mae_abs)
    mse_mean = np.mean(error_map_quad)
    rmse_mean = np.sqrt(mse_mean)
    # affichage
    """
    On veut une échelle comparative entre les images et avec l'error_map
        - pour l'affichage des images je prends l'échelle 0/v_max du film
        - pour l'error_map_abs je prends l'échelle 0/max(error_map_mae_abs)
        - pourm'erreur mae relative je prends l'échelle 0/10 (on affiche les erreurs relatives inférieures à 10x la valeur réelle)
        - pour l'error_map_quad je prends l'échelle 0/max(error_map_quad.max)
    """
    fig, ax = plt.subplots (1,3,sharey=True,figsize=(12,8))
    # Échelle commune pour les vitesses
    vmin_img, vmax_img = 0, v_max
    # Images de vitesse
    im0 = ax[0].imshow(img_tar, vmin=0, vmax=vmax_img)
    ax[0].set_title("Target")
    ax[0].set_axis_off()
    im1 = ax[1].imshow(img_pred, vmin=0, vmax=vmax_img)
    ax[1].set_title("Prediction")
    ax[1].set_axis_off()
    im2 = ax[2].imshow(error_map_mae_abs, vmin=0, vmax=error_max)
    ax[2].set_title("MAE prediction")
    ax[2].set_axis_off()
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label="m/s")
    stats_text = (
        f"erreur_abs_max = {error_max:.3f} m/s\n"
        f"erreur_abs_min = {error_min:.3f} m/s\n"
        f"erreur_abs_variance = {error_variance:.3f}\n"
        f"MAE moyenne = {mae_mean:.3f} m/s\n"
    )
    fig.suptitle(f"Prediction {i} — Erreurs absolues sur la prédiction de vitesse", fontsize=16, y=0.95, color='black')
    fig.text(0.01, 0.85, stats_text, fontsize=12, color='black', va='top')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_mae_abs_{i}.png'))
    # Cartes d'erreur
    fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(15,8))    
    im0 = ax[0].imshow(img_tar, vmin=0, vmax=vmax_img)
    ax[0].set_title("Target")
    ax[0].set_axis_off()
    im1 = ax[1].imshow(error_map_mae_abs, vmin=0, vmax=error_max)
    ax[1].set_title("MAE sur la vitesse")
    ax[1].set_axis_off()
    im4 = ax[2].imshow(error_map_mae_rel, cmap="coolwarm", vmin=0, vmax=2)
    ax[2].set_title("MAE relative")
    ax[2].set_axis_off()
    im5 = ax[3].imshow(error_map_quad_root, cmap="coolwarm", vmin=0, vmax=error_map_quad.max())
    ax[3].set_title("RMSE sur la vitesse (m²/s²)")
    ax[3].set_axis_off()
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label="m/s")
    fig.colorbar(im4, ax=ax[2], fraction=0.046, pad=0.04, label= "mae relative" )
    fig.colorbar(im5, ax=ax[3], fraction=0.046, pad=0.04, label="m/s")
        # ---- ANNOTATIONS ----
    # Ajoute les statistiques dans la figure
    stats_text = (
        f"MAE = {mae_mean:.3f} m/s\n"
        f"MSE = {mse_mean:.3f} (m/s)²\n"
        f"RMSE = {rmse_mean:.3f} m/s"
    )
    fig.suptitle(f"Prediction {i} — Erreurs sur la prédiction de vitesse", fontsize=16, y=0.95, color='black')
    fig.text(0.01, 0.85, stats_text, fontsize=12, color='black', va='top')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_mae-rmse_{i}.png'))
    plt.show()
    plt.close()
























