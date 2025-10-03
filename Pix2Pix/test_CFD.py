from pathlib import Path
import os
import numpy as np
from CFD_dataset_cas_4 import load_dataset
import matplotlib.pyplot as plt
import json
import pandas as pd
import tensorflow as tf
import keras


"""
script qui test la performance du meilleur modèle entrainé sur les 2/3 couples (input, taget) d'un film 
évaluation des performances sur un dataset de test = 20% de couples (input, target)

1- récupération du modèle ayant la loss la plus faible
2- loading G, D et best_criteria à partir du checkpoint enregistré par Ray


Dataset de train : Film de simulation CFD #1 (0 : 445)
Dataset de test : Film de simulation CFD #1 (790 : 990)

Output :
- image CFD d'entrée
- image CFD traduite
- image CFD réelle
- calcul des erreurs individuelles des predictions et sur l'ensemble des datasets
- courbe des pertes sur le dataset de test

"""

# dataset de test

dataset_name = "film_cfd_128_1.npy"
dataset_path = Path("/ccc/scratch/cont001/ocre/lebouchers/git/ImageMLProject/Datasets/") / dataset_name
dataset_path = dataset_path.resolve()
x_train, y_train, x_val, y_val, x_test, y_test = load_dataset(
        dataset_path, 
        offset=1, 
        split=(0.6, 0.2, 0.2))


# meilleur checkpoint
ray_results_dir = os.path.abspath(
    "./my_ray_results_cas_6/cas_6_run_1")
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

# modèle entrainé

generator = keras.models.load_model(os.path.join(
    checkpoint_path, "generator.keras"))
discrimintor = keras.models.load_model(os.path.join(
    checkpoint_path, "discriminator.keras"))
best_criteria_file = os.path.join(
    checkpoint_path, "training_state.json")


# évaluation du modèle

output_dir = os.path.abspath("./predictions_pix2pix_CFD_cas_6")
os.makedirs(output_dir, exist_ok=True)

mae_loss = tf.keras.losses.MeanAbsoluteError()

inp = x_test[10:11] # (1,128,128,1)
tar = y_test[10:11]

def prediction(initial_image, target_image):
    norm_initial_image = (initial_image/127.5)-1  # valeurs normalisées [-1,1]
    norm_target_image = (target_image/127.5)-1
    current_pred = generator(norm_initial_image, training=False)
    #pred = ((current_pred[0].numpy()+1) * 127.5) # on fait la prediction sur la première image de x_test
    pred = ((current_pred.numpy()+1) * 127.5) # on fait la prediction sur tout le x_test, valeurs [0,255]
    loss_mae_image = mae_loss(current_pred, norm_target_image).numpy() 
    return pred, loss_mae_image
    

pred, loss_mae = prediction(inp, tar)
# pred.shape (1,128,128,1)
# l'image = pred[0], (128,128,1)
print("MAE_loss :", loss_mae)
np.save(f"{output_dir}/premiere_pred.npy", pred[0])
image_10 = pred[0].astype(np.uint8) # format lisible par imshow
plt.imshow(image_10)
plt.savefig(os.path.join(output_dir, 'image_10.png'))



# Parcourir tout le dataset_test et enregister pred, initial et target

pred_dir = os.path.abspath(os.path.join(output_dir, "pred_dir"))
os.makedirs(pred_dir, exist_ok=True)

all_losses = []
for i  in range(len(x_test)):
    inp = x_test[i:i+1]
    tar = y_test[i:i+1]
    pred, loss = prediction(inp, tar)
    #np.save(f"{pred_dir}/y_pred_test{i}.npy", pred[0].astype(np.uint8))
    all_losses.append(loss)

mean_losses = np.mean(all_losses) # moyenne des pertes individuelles sur chacune des predictions

# sauvegarde des predictions sur le dataset de test
inp = x_test # parcourt tout le x_test
tar = y_test # parcourt tout le y_test
pred, loss = prediction(inp, tar) 
np.save(f"{output_dir}/y_pred_test_all.npy", pred) 
np.save(f"{output_dir}/loss_test_all.npy", loss)

dataset_mae = loss # perte calculée sur tout le dataset de test

# sauvegarde des predictions sur le dataset de train
inp_train = x_train # parcourt tout le x_train
tar_train = y_train # parcourt tout le y_train
pred_train, loss_train = prediction(inp_train, tar_train) 
np.save(f"{output_dir}/y_pred_train_all.npy", pred_train) 
np.save(f"{output_dir}/loss_train_all.npy", loss_train)

dataset_mae_train = loss_train # perte calculée sur tout le dataset de train

# sauvegarde des predictions sur le dataset de val
inp_val = x_val # parcourt tout le x_val
tar_val = y_val # parcourt tout le y
pred_val, loss_val = prediction(inp_val, tar_val)
np.save(f"{output_dir}/y_pred_val_all.npy", pred_val)
np.save(f"{output_dir}/loss_val_all.npy", loss_val)

dataset_mae_val = loss_val # perte calculée sur tout le dataset de val

# sauvegarde des predictions sur le dataset de test



# courbe des pertes
plt.figure(figsize=(10, 6))
plt.plot(all_losses, marker='o')
plt.xlabel("Index image")
plt.ylabel("Loss (MAE)")
plt.title("Courbe des pertes sur le dataset de test")
plt.grid(True)
plt.axhline(dataset_mae,color='red', linestyle='--', linewidth=2, label=f"perte_test: {dataset_mae:.4f}")
plt.axhline(dataset_mae_train,color='green', linestyle='--', linewidth=2, label=f"perte_train: {dataset_mae_train:.4f}")
plt.axhline(dataset_mae_val,color='blue', linestyle='--', linewidth=2, label=f"perte_val: {dataset_mae_val:.4f}")
plt.legend(loc='lower left', bbox_to_anchor=(0.2, 0.8), fontsize=9)
plt.savefig(os.path.join(output_dir, "courbe_des_pertes_test.png"))
plt.tight_layout()
plt.show()



# affichage des images


for i in range (0, x_test.shape[0], 50):
    input_i = x_test[i]
    target_i = y_test[i]
    pred_i = pred[i]
    error_map = np.abs(target_i.astype(np.float32) - pred_i.astype(np.float32))
    fig, ax = plt.subplots (1,4,sharey=True,figsize=(10, 4))
    ax[0].imshow(input_i)
    ax[0].set_title(f'Input {i}')
    ax[1].imshow(target_i)
    ax[1].set_title(f'Target {i}')
    ax[2].imshow(pred_i)
    ax[2].set_title(f'Prediction {i}')
    im=ax[3].imshow(error_map.squeeze(), cmap="hot")
    ax[3].set_title(f'Erreur {i}')
    fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
    fig.suptitle(f'Prediction Example {i}', fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f'fig_pred_{i}.png'))
    plt.show()
    plt.close()























