


"""

Script de lancement des entrainements avec Ray[tune]
Paramétrage de Ray et de la grille de recherche

"""


import tensorflow as tf
import os
from pathlib import Path
import json
from Pix2Pix.Datasets_Processing.dataset_CFD_processing import load_dataset
from Pix2Pix.Archi_Pix2Pix.architecture_model_GAN import Generator, Discriminator
from training import fit, calculate_gen_loss
import ray
from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Trainable
from ray.tune.tune_config import TuneConfig
from ray.air.config import RunConfig
from ray.train import CheckpointConfig



print("\n *** début du script python *** \n")

# on définit le dossier de stockage des outputs
output_dir = os.path.abspath("resultats_Pix2Pix")
os.makedirs(output_dir, exist_ok=True)

checkpoint_dir = os.path.abspath("./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)



class MyTrainable(Trainable):

    def setup(self, config):
        """
        Initialisation de l'entraînement.
        Cette méthode est appelée une fois au début de l'entraînement.
        """

        print("GPU disponibles : ", len(tf.config.list_physical_devices('GPU')),
              "GPUs trouvés :", tf.config.list_physical_devices('GPU'))

        # Initialisation des modèles et optimiseurs
        self.generator = Generator(
            config["image_size"], config["stride"], config["kernel_size"], config["OUTPUT_CHANNELS"])

        self.discriminator = Discriminator()

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"]
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"]
        )

        # Chargement du dataset
        self.train_dataset, self.val_dataset = load_dataset(BATCH_SIZE=config["BATCH_SIZE"], BUFFER_SIZE=400)

        # ce que l'on veut sauvegarder dans les checkpoints :
        self.checkpoint = tf.train.Checkpoint(
            generator=self.generator,
            discriminator=self.discriminator,
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
        )

        # Variables pour stocker les résultats
        self.gen_loss = None
        # self.gen_loss_values = []
        # self.step_values = []
        self.best_criteria = 1000.0

        # Nombre d'itérations d'entraînement (steps)
        self.nb_steps = config["nb_steps"]

        # initialisation du compteur des iterations de Ray
        self.global_step = 0


    def step(self):
        """
        Effectue une étape d'entraînement.
        Cette méthode est appelée à chaque itération.
        """

        # Appel de la fonction d'entraînement
        gen_loss, disc_loss = fit(
            self.generator, self.discriminator,
            self.generator_optimizer, self.discriminator_optimizer,
            self.train_dataset,
            self.val_dataset,
            steps=self.nb_steps,
            lambda_l1=self.config['lambda_l1'],
            start_step=self.global_step
        )

       # mise à jour du compteur d'itération
        self.global_step += self.nb_steps

       # Stockage des résultats
        self.gen_loss = gen_loss.numpy()

        # Calcul des loss d'intérêt
        train_gen_loss = calculate_gen_loss(self.generator, self.train_dataset)
        val_gen_loss = calculate_gen_loss(self.generator, self.val_dataset)

        # calcul du critère d'arrêt Jts
        #self.Jts = abs(self.train_gen_loss - self.val_gen_loss) + self.val_gen_loss


       # Retour des métriques à rapporter
        result = {
            "train_gen_loss": train_gen_loss,
            "val_gen_loss": val_gen_loss,
            "disc_loss": disc_loss.numpy(),
            "best_criteria": self.best_criteria,
            #"Jts": self.Jts
        }

        if train_gen_loss < self.best_criteria:
            print(
                f"Nouveau meilleur modèle : {train_gen_loss:.4f} < {self.best_criteria:.4f}")
            self.best_criteria = train_gen_loss
            result.update(should_checkpoint=True)

        return result

    def save_checkpoint(self, checkpoint_dir):
        """
        Sauvegarde le meilleur model :
          - generator 
          - le critère de performance
        """
        checkpoint_path = os.path.join(checkpoint_dir)
        os.makedirs(checkpoint_path, exist_ok=True)

        # sauvegarde du modèle
        self.generator.save(os.path.join(checkpoint_path, "generator.keras"))
        self.discriminator.save(os.path.join(
            checkpoint_path, "discriminator.keras"))

        with open(os.path.join(checkpoint_path, "best_criteria.json"), "w") as f:
            json.dump({"best_criteria": float(self.best_criteria)}, f)

        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """
        Recharge le meilleur model entrainé et son critère de performance
        """

        try:
            self.generator.load_model(os.path.join(
                checkpoint_path, "generator.keras")).expect_partial()
            self.discriminator.load_model(os.path.join(
                checkpoint_path, "discriminator.keras")).expect_partial()  # charge le model stocké
            print("succeeded, loaded model", checkpoint_path)
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            return  # Arrêter l'exécution si le modèle ne peut pas être chargé

        # recharge best_criteria
        best_criteria_file = os.path.join(
            checkpoint_path, "training_state.json")
        if os.path.exists(best_criteria_file):  # Vérifie si le fichier existe
            try:
                with open(best_criteria_file, "r") as f:
                    state = json.load(f)
                    self.best_criteria = state.get("best_criteria", None)
                    print(f"Critère chargé avec succès : {self.best_criteria}")
            except Exception as e:
                self.best_criteria = None
                print(f"Erreur lors du chargement du critère : {e}")
        else:
            self.best_criteria = None
            print("Avertissement : Aucun fichier 'training_state.json' trouvé.")


# grille des paramètres à tester = Search Space
config_search = {"image_size": 256,
                 "stride": tune.choice([2, 3, 4, 5, 6, 7, 8]),
                 "kernel_size": tune.choice([2, 3, 4, 5, 6, 7, 8]),
                 "OUTPUT_CHANNELS": 3,
                 "BATCH_SIZE": 1,
                 "nb_steps": 1000,
                 "lambda_l1": tune.uniform(0,200),
                 "learning_rate": tune.uniform(1e-5, 1e-2),
                 "beta_1": tune.uniform(0.5, 1.0)
                 }


# critères d'arrêt :

# arrête si la métrique ne s'améliore plus

stopper = TrialPlateauStopper(
    metric="best_criteria",  # métrique à surveiller
    mode="min",  # on veut minimiser la métrique
    std=1e-16,  # Seuil d'amélioration minimal, ici environ 3-5% de la valeur de la loss
    num_results=15  # nbre d'iterations avant de checker le plateau
)

# planificateur ASHA pour accélérer l'optimisation

scheduler = ASHAScheduler(
    metric="best_criteria",
    mode="min",
    max_t=800,
    grace_period=800   # nbre min d'iterations avant de checker le plateau
)


# configuration de tune

tune_config = TuneConfig(
    num_samples=4,
    max_concurrent_trials=4,
    scheduler=scheduler
)

# configuration des checkpoints

checkpoint_config = CheckpointConfig(
    checkpoint_frequency=0,
    num_to_keep=1,
    checkpoint_score_attribute="gen_loss",
    checkpoint_score_order="min",
    checkpoint_at_end=False
)


# configuration de l'exécution

run_config = RunConfig(
    name="run_facades_dataset",
    storage_path=os.path.abspath("./my_ray_results_facades_dataset"),
    stop=stopper,  # ou stop={"training_iteration": 10}
    checkpoint_config=checkpoint_config
)


if __name__ == "__main__":
    # Ce que tu veux faire quand tu utilises le script directement
    # C'est à dire python Trainable.py
    # Ca evite de lancer tout le script quand tu fais un import
    print("********* initialisation de Ray *******************")
    ray.init(address="auto", include_dashboard=False)
    print(ray.cluster_resources())
    print("initialisation de Ray :", ray.is_initialized())

    print("Chargement des datasets")
    dataset_name = "facades"
    dataset_path = Path("./git/ImageMLProject/Datasets/") / dataset_name
    dataset_path = dataset_path.resolve()

    trainable = tune.with_parameters(
        MyTrainable
    )
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 48, "gpu": 1})
    print("Lancement de l'optimisation avec Tuner")
    
    # récupération si entrainement interrompu
    dir_path=os.path.abspath("./my_ray_results_facades_dataset/run_facades_dataset")
    if tune.Tuner.can_restore(dir_path):
        print(f"Restauration de l'entrainement depuis {dir_path}")
        tuner=tune.Tuner.restore(
            dir_path,
            trainable=trainable_with_resources,
            resume_errored=True
        )
    else :
        tuner = Tuner(
            trainable_with_resources,
            param_space=config_search,
            tune_config=tune_config,
            run_config=run_config,
        )
    results = tuner.fit()
    ray.shutdown()


