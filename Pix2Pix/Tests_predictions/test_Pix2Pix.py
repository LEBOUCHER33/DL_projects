"""
Script to test the trained Pix2Pix model with the best parameters
    -load the best trained model
    -load the test dataset
    -generate and save predictions

"""


import tensorflow as tf
from pathlib import Path
import os
import time
from PIL import Image
import numpy as np
import json
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt



# 1- processing du dataset de test

dataset_name = "facades"
dataset_path = Path("./git/ImageMLProject/Datasets/") / dataset_name
dataset_path = dataset_path.resolve()
test_dataset=tf.data.Dataset.list_files(str(dataset_path / 'test/*.jpg'))

BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

def load(image_file):
  image = tf.io.read_file(image_file)  
  image = tf.io.decode_jpeg(image)  
  w = tf.shape(image)[1]   
  w = w // 2 
  input_image = image[:, w:, :]  
  real_image = image[:, :w, :]   
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)
  return input_image, real_image


def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image


def resize(input_image, real_image, height, width):
	input_image = tf.image.resize(input_image, [height, width],  
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 
	real_image = tf.image.resize(real_image, [height, width],  
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
	return input_image, real_image


def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image

test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)


# 2- loading du meilleur checkpoint / modele entrainé

ray_results_dir = os.path.abspath(
    "./my_ray_results_facades_dataset/hyperparam_tuning_facades_dataset")
os.makedirs(ray_results_dir, exist_ok=True)
print(ray_results_dir)

dataframes = []

for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        results_file = os.path.join(trial_path, "result.json")
        params_file = os.path.join(trial_path, "params.json")
        if os.path.isfile(params_file) and os.path.isfile(results_file):
            with open(params_file, "r") as f:
                params = json.load(f)
            with open(results_file, "r") as f:
                try:
                    result_json = pd.read_json(results_file, lines=True)
                    result_json["trial_id"] = trial_dir
                    dataframes.append(result_json)
                except (json.JSONDecodeError, TypeError) as e:
                    print("no result")
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

# meilleur modèle entrainé
generator = keras.models.load_model(os.path.join(
    checkpoint_path, "generator.keras"))
discrimintor = keras.models.load_model(os.path.join(
    checkpoint_path, "discriminator.keras"))
best_criteria_file = os.path.join(
    checkpoint_path, "training_state.json")

 



# 3- évaluation des performances visuelles de predictions du modèle

output_dir = os.path.abspath("./predictions_pix2pix_facades_dataset")
os.makedirs(output_dir, exist_ok=True)

global_start = time.perf_counter()

for i, (inp, tar) in enumerate (test_dataset.take(5)):
  start = time.perf_counter()
  prediction = generator(inp, training=True)
  print(f'Time taken for prediction with best_model : {time.perf_counter()-start:.2f} sec\n')
  inp_image = ((inp[0].numpy() * 0.5 + 0.5)*255).astype(np.uint8)
  tar_image = ((tar[0].numpy()* 0.5 + 0.5) * 255).astype(np.uint8)
  pred_image = ((prediction[0].numpy()* 0.5 + 0.5) * 255).astype(np.uint8)
  Image.fromarray(inp_image).save(os.path.join(output_dir, f"best_model_input_{i}.png"))
  Image.fromarray(tar_image).save(os.path.join(output_dir, f"best_model_target_{i}.png"))
  Image.fromarray(pred_image).save(os.path.join(output_dir, f"best_model_prediction_{i}.png"))
  print(f'Time taken for saving : {time.perf_counter()-start:.2f} sec\n')
  fig, ax = plt.subplots (1,3,sharey=True,figsize=(10, 4))
  ax[0].imshow(inp_image)
  ax[0].set_title(f'Input {i}')
  ax[1].imshow(tar_image)
  ax[1].set_title(f'Target {i}')
  ax[2].imshow(pred_image)
  ax[2].set_title(f'Prediction {i}')
  fig.suptitle(f'Prediction Example {i}', fontsize=14)
  fig.tight_layout()
  plt.savefig(os.path.join(output_dir, f'fig_pred_{i}.png'))
  plt.show()
  plt.close()

print(f'Temps total pour 5 images : {time.perf_counter() - global_start}')

