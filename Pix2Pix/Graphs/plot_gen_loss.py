import os
import matplotlib.pyplot as plt
import json
import pandas as pd

"""
Script d’analyse des entraînements avec hyperparamètres :

- Affiche les courbes d’évolution de la métrique 'best_criteria'
- Met en évidence les meilleurs essais (top 10%) en bleu

"""

# Répertoires
output_dir = os.path.abspath("./graph_results_cas_6")
os.makedirs(output_dir, exist_ok=True)

ray_results_dir = os.path.abspath("./my_ray_results_cas_6/cas_6_run_1")
os.makedirs(ray_results_dir, exist_ok=True)

# Chargement des fichiers result.json
dataframes = []

for trial_dir in os.listdir(ray_results_dir):
    trial_path = os.path.join(ray_results_dir, trial_dir)
    if os.path.isdir(trial_path):
        results_file = os.path.join(trial_path, "result.json")
        params_file = os.path.join(trial_path, "params.json")
        if os.path.isfile(params_file) and os.path.isfile(results_file):
            try:
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
print(result_df.shape)
print(result_df.head())
print(result_df.columns.tolist())
print("Nombre d'essais uniques :", result_df["trial_id"].nunique())

# Détermination des 10 % meilleurs essais
df_last = result_df.groupby("trial_id").tail(1)  # Dernier point par trial
df_best_sorted = df_last.sort_values("best_criteria")
trial_ids = df_best_sorted["trial_id"].tolist()
top_10_percent = max(1, int(len(df_best_sorted) * 0.1))
df_top = df_best_sorted.head(top_10_percent)
top_trial_ids = df_top["trial_id"].tolist()

# Tracé des courbes
plt.figure(figsize=(16, 8))

for trial in result_df.trial_id.unique():
    trial_data = result_df[result_df["trial_id"] == trial]
    
    if "training_iteration" in trial_data.columns and "best_criteria" in trial_data.columns:
        step_list = trial_data["training_iteration"].tolist()
        best_criteria_list = trial_data["best_criteria"].tolist()
        
        color = "blue" if trial in top_trial_ids else "lightgray"
        linewidth = 2 if trial in top_trial_ids else 1
        alpha = 1.0 if trial in top_trial_ids else 0.5
        #label = f"Trial {trial}" if trial in top_trial_ids else None
        
        plt.plot(step_list, best_criteria_list, color=color,
                 linewidth=linewidth, alpha=alpha)
        # valeur min de la loss identifiée
        if trial in top_trial_ids:
            min_idx = best_criteria_list.index(min(best_criteria_list))
            min_step = step_list[min_idx]
            min_val = best_criteria_list[min_idx] 
            plt.scatter(min_step, min_val, color=color, s=50, zorder=5, edgecolors="black")
            plt.annotate(f"{min_val:.2e}",   
                         (min_step, min_val),
                         textcoords="offset points",
                         xytext=(5,5),       # petit décalage
                         fontsize=9,
                         color=color,
                         fontweight="bold")


# Mise en forme
plt.xlabel("Iterations")
plt.ylabel("Total_loss")
plt.ylim(1e-3, 1e-0)
plt.yscale("log")
plt.title("Évolutions des pertes du générateur")
plt.tight_layout()

from matplotlib.lines import Line2D

# Légende personnalisée (couleur uniquement)
custom_lines = [
    Line2D([0], [0], color="blue", lw=2, label=f"Top 10% essais ({len(top_trial_ids)} essais)"),
    Line2D([0], [0], color="lightgray", lw=2, label=f"Autres essais ({len(trial_ids)} essais)"),
]

plt.legend(handles=custom_lines, loc="upper right")
plt.savefig(os.path.join(output_dir, "graph_training_criteria_cas_6.png"))
plt.show()


