

"""
Script qui configure la recherche d'hyperparamètres avec Ray[tune]

Paramétrage de Ray (ressources / checkpoints / stopper) et de la grille de recherche

Dataset de CFD avec des paires inp/tar mélangées aléatoirement

"""



import tensorflow as tf
import os
from pathlib import Path
import json
from Pix2Pix.training import fit, calculate_gen_loss
from Pix2Pix.Datasets_Processing.dataset_CFD import load_dataset, get_tf_dataset, resize_and_normalize
from Pix2Pix.Archi_Pix2Pix.architecture_model_GAN import Generator, Discriminator
import ray
from ray import tune
from ray.tune.tuner import Tuner
from ray.tune.stopper import TrialPlateauStopper
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Trainable
from ray.tune.tune_config import TuneConfig
from ray.tune import RunConfig
from ray.tune import CheckpointConfig



checkpoint_dir = os.path.abspath("./checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)



class MyTrainable(Trainable):


    def setup(self, config, dataset_path):
#    def setup(self, config, inputs, targets, train_idx, test_idx, val_idx):
        """
        Initialisation de l'entraînement.
        Cette méthode est appelée une fois au début de l'entraînement.
        """

        print("GPU disponibles : ", len(tf.config.list_physical_devices('GPU')),
              "GPUs trouvés :", tf.config.list_physical_devices('GPU'))

        # Initialisation des modèles et optimiseurs
        self.generator = Generator(
            config["image_size"], config["stride"], config["kernel_size"], config["OUTPUT_CHANNELS"])

        self.discriminator = Discriminator(image_size=config["image_size"])

        self.generator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"]
        )
        self.discriminator_optimizer = tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"],
            beta_1=config["beta_1"]
        )

        self.generator.compile(loss=tf.keras.losses.MeanAbsoluteError())

        # Chargement du dataset
        # Sélectionne les données, redimensionne et normalise
        inputs, targets, train_idx, test_idx, val_idx = load_dataset(dataset_path, offset=1)
        x_train, y_train = resize_and_normalize(inputs[train_idx], targets[train_idx])
        x_val, y_val = resize_and_normalize(inputs[val_idx], targets[val_idx])

        self.train_dataset = get_tf_dataset(x_train, y_train, BATCH_SIZE=config["BATCH_SIZE"])
        self.val_dataset = get_tf_dataset(x_val, y_val, BATCH_SIZE=1, shuffle=False)

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
        self.best_criteria = 10000.0

        # Nombre d'itérations d'entraînement (steps)
        self.nb_steps = config["nb_steps"]

        # initialisation du compteur des iterations de Ray
        self.global_step = 0


        # période de chauffe
        # self.burnin=True

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
       
       # on peut définir un nouveau score de performance pour éviter le sur-apprentissage
        """
        Un indicateur de performance du modèle serait d'avoir une perte minimale du generateur sur le dataset de validation
        Il faut néanmoins éviter que la perte du générateur sur la validation soit supérieure à celle du dataset d'entrainement
        On peut donc définir un différentiel : max (0, val_gen_loss - train_gen_loss) 
        L'objectif est d'éviter que ce différentiel atteigne une valeur trop importante = signe de sur-apprentissage
        On peut ainsi créer un critère combiné avec les deux éléments :
        score = val_gen_loss + alpha * max (0, val_gen_loss - train_gen_loss) avec alpha qui pondère la sévérité du sur-apprentissage
        ainsi :
            - diff faible et val_gen_loss grand = sous-apprentissage
            - diff faible et val_gen_loss petit = généralisation  : meilleur score
            - diff fort et val_gen_loss grand = sur-apprentissage
        """
        alpha = 0.5
        diff = max(0, val_gen_loss - train_gen_loss)
        score = val_gen_loss + alpha * diff
        

       # Retour des métriques à rapporter
        result = {
            "train_gen_loss": train_gen_loss,
            "val_gen_loss": val_gen_loss,
            "disc_loss": disc_loss.numpy(),
            "best_criteria": self.best_criteria,
            "score": score
        }

        if score < self.best_criteria:
            print(
            f"Nouveau meilleur modèle : {score:.4f} < {self.best_criteria:.4f} "
            f"(val={val_gen_loss:.4f}, train={train_gen_loss:.4f}, diff={diff:.4f})"
            )
            self.best_criteria = score
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
config_search = {"image_size": 128,
                 "stride": tune.choice([2, 3, 4, 5, 6, 7, 8]),
                 "kernel_size": tune.choice([2, 3, 4, 5, 6, 7, 8]),
                 "OUTPUT_CHANNELS": 1,
                 "BATCH_SIZE": 1,
                 "nb_steps": 1000,
                 "lambda_l1": tune.uniform(0,200),
                 "learning_rate": tune.loguniform(1e-3, 4e-3),
                 "beta_1": tune.uniform(0.5, 1.0)
                 }


# critères d'arrêt :

# arrête si la métrique ne s'améliore plus

stopper = TrialPlateauStopper(
    metric="score",  # métrique à surveiller
    mode="min",  # on veut minimiser la métrique
    std=1e-16,  # Seuil d'amélioration minimal, ici environ 3-5% de la valeur de la loss
    num_results=15  # nbre d'iterations avant de checker le plateau
)

# planificateur ASHA pour accélérer l'optimisation

scheduler = ASHAScheduler(
    metric="score",
    mode="min",
    max_t=500,
    grace_period=500   # nbre min d'iterations avant que l'algorithme ASHA agisse
)


# configuration de tune

tune_config = TuneConfig(
    num_samples=1000,
    max_concurrent_trials=4,
    scheduler=scheduler
)

# configuration des checkpoints

checkpoint_config = CheckpointConfig(
    checkpoint_frequency=0,
    num_to_keep=1,
    checkpoint_score_attribute="score",
    checkpoint_score_order="min",
    checkpoint_at_end=False
)


# configuration de l'exécution

run_config = RunConfig(
    name="cas_8_run_1",
    storage_path=os.path.abspath("./my_ray_results_cas_8"),
    stop=stopper,  # ou stop={"training_iteration": 10}
    checkpoint_config=checkpoint_config
)

# lancement de l'optimisation avec Tuner


if __name__ == "__main__":
    # Ce que tu veux faire quand tu utilises le script directement
    # C'est à dire python Trainable.py
    # Ca evite de lancer tout le script quand tu fais un import
    print("********* initialisation de Ray *******************")
    ray.init(address="auto", include_dashboard=False)
    print(ray.cluster_resources())
    print("initialisation de Ray :", ray.is_initialized())

    print("Chargement des datasets")
    dataset_name = "film_3.npy"
    dataset_path = Path("./git/ImageMLProject/Datasets/CFD_Dataset/") / dataset_name
    dataset_path = dataset_path.resolve()

    trainable = tune.with_parameters(
        MyTrainable,
        dataset_path=dataset_path
    )
    trainable_with_resources = tune.with_resources(trainable, {"cpu": 48, "gpu": 1})
    # récupération si entrainement interrompu
    restore_path=os.path.join(
        os.path.abspath("./my_ray_results_cas_8/cas_8_run_1")
    )
    if tune.Tuner.can_restore(restore_path):
        print(f"Restauration de l'entrainement depuis {restore_path}")
        tuner=tune.Tuner.restore(
            restore_path,
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
    


