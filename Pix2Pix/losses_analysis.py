
"""
Script pour analyser les performances des différents essais d'entraînement
    -chargement des fichiers de résultats
    -comparaison des performances
    -tracé des courbes de perte d'entrainement et de validation

"""


# import des librairies
import os
import matplotlib.pyplot as plt
import json
import pandas as pd


# Répertoires

# répertoire de sauvegarde des graphes
output_dir = os.path.abspath("./graph_results_facades")
os.makedirs(output_dir, exist_ok=True)
# répertoire des résultats de Ray Tune
ray_results_dir = os.path.abspath("./my_ray_results_facades_dataset/hyperparam_tuning_facades_dataset")
os.makedirs(ray_results_dir, exist_ok=True)

# Chargement des fichiers result.json
dataframes = []

# Pour chaque réseau testé
for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        # On récupère le fichier result.json
        results_file = os.path.join(trial_path, "result.json")
        if os.path.isfile(results_file):
            try:
                # On le lit et on l'ajoute à la liste
                result_json = pd.read_json(results_file, lines=True)
                result_json["trial_id"] = trial_dir
                dataframes.append(result_json)
            except (json.JSONDecodeError, TypeError):
                print(f"Fichier illisible pour {trial_dir}")
                continue


if not dataframes:
    raise ValueError("Aucun fichier de résultats valide trouvé.")


# Concaténation des données
result_df = pd.concat(dataframes, ignore_index=True)
print(result_df.shape,result_df.head(),result_df.columns.tolist())

n_trials=result_df["trial_id"].nunique()
print("Nombre d'essais uniques :", n_trials)
all_trial_ids = result_df["trial_id"].unique().tolist()

# Selection du meilleur pt / trial
best_per_trial = result_df.loc[result_df.groupby("trial_id")["best_criteria"].idxmin()]
df_best_sorted = best_per_trial.sort_values("best_criteria")
trial_ids = best_per_trial["trial_id"].tolist()

# top 10%
top_10_percent = max(1, int(len(df_best_sorted) * 0.1))
df_top = df_best_sorted.head(top_10_percent)
top_trial_ids = df_top["trial_id"].tolist()
print(f"{len(top_trial_ids)} meilleurs essais sélectionnés sur {len(df_best_sorted)}.")



# Tracé des courbes
def smooth(y, window_size=1):
    """
    _Summary_ : fonction qui lisse une série temporelle par une moyenne glissante
    _Args_ :  y (série temporelle),
              window_size (taille de la fenêtre de lissage)
    _Returns_ : la série lissée (courbe plus lisse, visuellement plus agréable)
    """
    return y.rolling(window=window_size, min_periods=1, center=True).mean()

# ========================
# Fonction de tracé simple
# ========================
def plot_losses(df, trial_ids, linewidth=1, alpha=0.6):
    """
    _Summary_ : fonction qui trace les courbes de perte d'entrainement et de validation
    _Args_ :  df (dataframe des résultats),
              trial_ids (liste des IDs des essais à tracer),
              linewidth (épaisseur des lignes),
              alpha (transparence des lignes)
    _Returns_ : None
    _Side Effects_ : affiche et sauvegarde les graphes dans le répertoire output_dir
    """
    # Pour gérer la légende
    legend_done = {"train": False, "val": False, "disc": False}
    # Pour chaque essai
    for i, trial in enumerate(trial_ids):
        trial_data = df[df["trial_id"] == trial]
        if "training_iteration" in trial_data.columns:
            x = trial_data["training_iteration"]
            if "train_gen_loss" in trial_data.columns:
                plt.plot(
                    x,
                    smooth(trial_data["train_gen_loss"]),
                    color=f"C{i}",
                    linestyle="-",
                    linewidth=linewidth,
                    alpha=alpha,
                    label="train_loss" if not legend_done["train"] else None
                )
                legend_done["train"] = True
            if "val_gen_loss" in trial_data.columns:
                plt.plot(
                    x,
                    smooth(trial_data["val_gen_loss"]),
                    color=f"C{i}",
                    linestyle="--",
                    linewidth=linewidth,
                    alpha=alpha,
                    label="val_loss" if not legend_done["val"] else None
                )
                legend_done["val"] = True
            #if "disc_loss" in trial_data.columns:
            #    plt.plot(
            #        x,
            #        smooth(trial_data["disc_loss"]),
            #        color=f"C{i}",
            #        linestyle=":",
            #        linewidth=linewidth,
            #        alpha=alpha,
            #        label="disc_loss" if not legend_done["disc"] else None
            #    )
            #    legend_done["disc"] = True

# ========================
# Tracé principal
# ========================
plt.figure(figsize=(16, 8))
plot_losses(result_df, trial_ids, linewidth=0.5, alpha=0.5)
# Mise en forme
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.yscale("log")
plt.legend()
plt.title(f"Évolution des pertes générateur lissées ({len(trial_ids)} essais)")
plt.tight_layout()
# Sauvegarde
plt.savefig(os.path.join(output_dir, "graph_training_losses_cas_4_total_trials.png"))
plt.show()

plt.figure(figsize=(16, 8))
plot_losses(result_df, top_trial_ids,linewidth=2, alpha=1)
plt.xlabel("Itérations")
plt.ylabel("Perte")
plt.yscale("log")
plt.title(f"Évolution des pertes générateur lissées (Top 10% : {len(top_trial_ids)} essais)")
plt.legend()
plt.tight_layout()
# Sauvegarde
plt.savefig(os.path.join(output_dir, "graph_training_losses_cas_4_top_trials.png"))
plt.show()

