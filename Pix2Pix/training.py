

"""
Script pour l'entrainement du modèle "pix2pix"
    -fonction de perte
    -fonction d'entrainement
    -fonction de sauvegarde et visualisation des résultats

"""

# import des librairies

import tensorflow as tf
import os
import matplotlib.pyplot as plt



# 1- Calcul des fonctions de perte

# on utilise la BinaryCrossentropy pour la perte du discriminateur et du générateur
loss_object=tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target, lambda_l1=100):
  """
  _Summary_ : fonction qui calcule la perte du générateur
  _Args_ :  disc_generated_output (sortie du discriminateur pour les images générées), 
            gen_output (images générées par le générateur), 
        target (images réelles), 
        lambda_l1 (poids de la perte L1)
  _Returns_ : la perte totale du générateur
  """
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  total_gen_loss = gan_loss + (lambda_l1 * l1_loss)
  return total_gen_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  """
  _Summary_ : fonction qui calcule la perte du discriminateur
  _Args_ :  disc_real_output (sortie du discriminateur pour les images réelles), 
            disc_generated_output (sortie du discriminateur pour les images générées)
  _Returns_ : la perte totale du discriminateur = perte sur les images réelles + perte sur les images générées
  """
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
  total_disc_loss = real_loss + generated_loss
  return total_disc_loss



# 2- Fonctions d'entrainement du modèle


@tf.function
def train_step(generator, discriminator, generator_optimizer, discriminator_optimizer, input_image, target, lambda_l1):
  """
  _Summary_ : fonction qui effectue une étape d'entrainement du modèle
  _Args_ :  generator (modèle du générateur), 
            discriminator (modèle du discriminateur),
            generator_optimizer (optimiseur du générateur),
            discriminator_optimizer (optimiseur du discriminateur),
            input_image (image d'entrée),
            target (image réelle),
            lambda_l1 (poids de la perte L1)
  _Returns_ : la perte totale du générateur et du discriminateur
  """
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)  # generer une image
    disc_real_output = discriminator([input_image, target], training=True) # classifier l'image réelle
    disc_generated_output = discriminator([input_image, gen_output], training=True) # classifier l'image générée
    
# calcul des pertes :
    gen_total_loss = generator_loss(disc_generated_output, gen_output, target, lambda_l1)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    print("Valeur de gen_total_loss :", gen_total_loss)
    print("Valeur de disc_loss :", disc_loss)
# mise à jour des poids
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  return gen_total_loss, disc_loss



output_dir = os.path.abspath("./resultats_CFD_2")

# sauvegarde et visualisation des résultats du modèle
def generate_images(model, input_image, target):
    """
    _Summary_ : fonction qui génère et sauvegarde les images produites par le générateur
    _Args_ :  model (modèle du générateur),
              input_image (image d'entrée),
              target (image réelle)
    _Returns_ : l'image générée par le modèle"""
    prediction = model(input_image, training=False)
    display_list = [input_image[0], target[0], prediction[0]]
    title = ['Input Image', 'Target Image', 'Generated Image']
    for i in range(3):
      plt.close('all')
      plt.figure()
      plt.imshow(display_list[i] * 0.5 + 0.5) # normalise de [-1;1] à [0;1]
      plt.axis('off')
      plt.savefig(os.path.join(output_dir, f"{title[i]}_{i}"),bbox_inches='tight', pad_inches=0)
    return prediction

"""
# mesure de la perte L1 sur le dataset de validation
def validation_step(generator, val_ds, lambda_l1):
    for val_input, val_target in val_ds:
        prediction = generator(val_input, training=False)
        l1_loss = tf.reduce_mean(tf.abs(val_target - prediction))
        gen_total_loss = lambda_l1 * l1_loss  
    return gen_total_loss.numpy()
"""


# on va utiliser cette fonction de perte pour améliorer le modèle

def fit(generator, discriminator, generator_optimizer, discriminator_optimizer, train_ds, val_ds, steps, lambda_l1,start_step=0):
  """
  _Summary_ : fonction qui entraine le modèle sur un certain nombre d'étapes
  _Args_ :  generator (modèle du générateur), 
            discriminator (modèle du discriminateur),
            generator_optimizer (optimiseur du générateur),
            discriminator_optimizer (optimiseur du discriminateur),
            train_ds (dataset d'entrainement),
            val_ds (dataset de validation),
            steps (nombre d'étapes d'entrainement),
            lambda_l1 (poids de la perte L1),
            start_step (étape de départ, utile pour la reprise d'entrainement)
  _Returns_ : la perte totale du générateur et du discriminateur à la dernière étape"""
# boucle d'entrainement : on prend un batch ('steps' images) du dataset d'entrainement et on effectue une étape d'entrainement
  for local_step, (input_image, target) in enumerate(train_ds.repeat().take(steps)):
    gen_total_loss, disc_loss = train_step(generator, discriminator, generator_optimizer, discriminator_optimizer,input_image, target, lambda_l1)

  return gen_total_loss, disc_loss

# fonction de perte MAE
mae_loss = tf.keras.losses.MeanAbsoluteError()

def calculate_gen_loss(generator, dataset):
    """
    _Summary_ : fonction qui calcule la perte MAE du générateur sur un dataset donné
    _Args_ :  generator (modèle du générateur),
              dataset (dataset sur lequel calculer la perte)
    _Returns_ : la perte MAE moyenne du générateur sur le dataset 
    """
    total_loss = 0.0
    count = 0
    for input_image, target in dataset:
        prediction = generator(input_image, training=False)
        loss = mae_loss(target, prediction)
        total_loss += loss.numpy()
        count += 1

    return total_loss / count if count > 0 else 0



def calculate_full_gen_loss(generator, discriminator, dataset, lambda_l1=100):
    """
    _Summary_ : fonction qui calcule la perte totale du générateur (perte GAN + perte L1) sur un dataset donné
    _Args_ :  generator (modèle du générateur),
              discriminator (modèle du discriminateur),
              dataset (dataset sur lequel calculer la perte),               
              lambda_l1 (poids de la perte L1)
    _Returns_ : la perte totale moyenne du générateur sur le dataset 
    """
    total_loss = 0.0
    count = 0

    for input_image, target in dataset:
        prediction = generator(input_image, training=False)
        disc_output = discriminator([input_image, prediction], training=False)
        
        gan_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(disc_output), disc_output)
        l1_loss = tf.reduce_mean(tf.abs(target - prediction))
        
        gen_total_loss = gan_loss + lambda_l1 * l1_loss
        total_loss += gen_total_loss.numpy()
        count += 1

    return total_loss / count if count > 0 else 0


















