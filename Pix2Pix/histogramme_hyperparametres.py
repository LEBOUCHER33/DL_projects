

import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np

"""

script qui permet d'afficher les histogrammes des résultats d'entrainements de chacun des hyperparamètres testés

objectif :

- on récupère le fichier result.json produit par Ray
- on le convertit en DataFrame
- on trie le dataframe par ordre croissant de best_criteria
- on récupère les 10% meilleurs
- pour chacun des hyperparamètres testés on affiche un histogramme de ses différentes valeurs

"""

output_dir = os.path.abspath("./graph_results_cas_6")
os.makedirs(output_dir, exist_ok=True)
print(output_dir)
ray_results_dir = os.path.abspath("./my_ray_results_cas_6/cas_6_run_1")
os.makedirs(ray_results_dir, exist_ok=True)
print(ray_results_dir)
dataframes=[]

for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        results_file = os.path.join(trial_path, "result.json")
        params_file = os.path.join(trial_path, "params.json")        
        if os.path.isfile(params_file) and os.path.isfile(results_file):
            with open(params_file, "r") as f:
                params = json.load(f)
            with open(results_file, "r") as f:
                try :
                    result_json=pd.read_json(results_file, lines=True)
                    result_json["trial_id"] = trial_dir
                    dataframes.append(result_json)                    
                except (json.JSONDecodeError, TypeError) as e:
                    print ("no result")
                    continue
if not dataframes:
    raise ValueError("Aucun fichier de résultats valide trouvé.")

# je réunis les trials dans un dataframe
result_df=pd.concat(dataframes,ignore_index=True)

# je recupère uniquement la métrique la plus performante / trial
df_best = result_df.loc[result_df.groupby("trial_id")["best_criteria"].idxmin()]
print(df_best.shape[0])
n_trials=result_df["trial_id"].nunique()

# je trie par ordre ascendant
df_best_sorted = df_best.sort_values("best_criteria")

# je récupère les 10% meilleurs
top_10_percent = int(len(df_best_sorted) * 0.1)
#df_top = df_best_sorted.head(top_10_percent)
df_top = df_best_sorted.head(10)
print (df_top.shape, df_top.head(5))

config_all_df = pd.json_normalize(df_best_sorted["config"])
config_top_df = pd.json_normalize(df_top["config"])

nb_training_best=df_top.shape[0]



# 7- je trace un histogramme pour chacun des hyperparamètres 

import seaborn as sns

config_all_df = pd.json_normalize(result_df["config"])
config_top_df = pd.json_normalize(df_top["config"])


# Colonnes à exclure
excluded_cols = ["image_size", "OUTPUT_CHANNELS","BATCH_SIZE","nb_steps"]

# Filtrage des colonnes à afficher
columns_to_plot = [col for col in config_top_df.columns if col not in excluded_cols]

import math
num_plots = len(columns_to_plot)
cols = 3
rows = math.ceil(num_plots / cols)
fig, axes = plt.subplots(rows, cols, figsize=(25,12))
axes = axes.flatten()

for i, col in enumerate(columns_to_plot):
    ax = axes[i]
    
    all_values = config_all_df[col].dropna()
    top_values = config_top_df[col].dropna()
    
    min_val = min(all_values.min(), top_values.min())
    max_val = max(all_values.max(), top_values.max())
    bins = np.linspace(min_val, max_val, 20)

    ax.hist(all_values, bins=bins, alpha=0.3, label="All trials", color="gray", edgecolor="black", density=True)
    ax.hist(top_values, bins=bins, alpha=0.7, label="Top 10%", color="blue", edgecolor="black", density=True)

    if all_values.nunique() > 1:
        sns.kdeplot(all_values, ax=ax, color="gray", linestyle="--", linewidth=1.5, label="KDE All")
    if top_values.nunique() > 1:
        sns.kdeplot(top_values, ax=ax, color="blue", linewidth=2, label="KDE Top 10%")

    ax.set_title(col, fontsize=10, color="orange")
    ax.set_xlabel(None)
    ax.set_ylabel("Density")
    ax.grid(False)
    ax.legend()

# Supprimer les axes inutilisés
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.suptitle(f"Distribution des valeurs des hyperparamètres ({n_trials} essais, {nb_training_best} tops)",
             fontsize=12, y=0.99, color="blue")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "subplot_histogrammes_superposes_kde.png"))
plt.show()













