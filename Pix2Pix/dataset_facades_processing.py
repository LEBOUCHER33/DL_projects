

# import des librairies
from pathlib import Path
import tensorflow as tf


"""
Script pour le chargement et la préparation du dataset "facades"
    -chargement des images
    -redimensionnement
    -normalisation
    -séparation en dataset d'entrainement et de validation
    -fonction pour charger le dataset
"""

# dataset d'entrainement et préparation des images
dataset_name = "facades"
path = Path("./git/ImageMLProject/Datasets/") / dataset_name
path = path.resolve()


IMG_WIDTH = 256
IMG_HEIGHT = 256

# preparation des fonctions pour le chargement et la préparation des images

def load(image_file):
  """
  _Summary_ : fonction qui charge une image et la divise en deux images (input et real)
  _Args_ : path de l'image
  _Returns_ : les deux images (input et real)
  """
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
  """
  _Summary_ : fonction qui normalise les images entre -1 et 1
  _Args_ : les deux images (input et real)
  _Returns_ : les deux images normalisées
  """
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1
  return input_image, real_image


def resize(input_image, real_image, height, width):
   """
   _Summary_ : fonction qui redimensionne les images
   _Args_ : les deux images (input et real), la hauteur et la largeur
   _Returns_ : les deux images redimensionnées
   """
   input_image = tf.image.resize(input_image, [height, width],  
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   real_image = tf.image.resize(real_image, [height, width],  
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
   return input_image, real_image


@tf.function()  
def random_jitter(input_image, real_image):
  """
  _Summary_ : fonction qui applique des transformations aléatoires aux images
  _Args_ : les deux images (input et real)
  _Returns_ : les deux images transformées
  """
  input_image, real_image = resize(input_image, real_image, 286, 286)
  input_image, real_image = random_crop(input_image, real_image)
  if tf.random.uniform(()) > 0.5:
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)
  return input_image, real_image

def random_crop(input_image, real_image):
  """
  _Summary_ : fonction qui applique une coupe aléatoire aux images
  _Args_ : les deux images (input et real)
  _Returns_ : les deux images coupées
  """
  stacked_image = tf.stack([input_image, real_image], axis=0)  
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3]) 
  return cropped_image[0], cropped_image[1] 

def load_image_val(image_file):
  """
  _Summary_ : fonction qui charge et prépare les images de validation
  _Args_ : path de l'image
  _Returns_ : les deux images (input et real) redimensionnées et normalisées
  """
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image


def load_image_train(image_file):
  """
  _Summary_ : fonction qui charge et prépare les images d'entrainement
  _Args_ : path de l'image
  _Returns_ : les deux images (input et real) redimensionnées, normalisées et avec des transformations aléatoires
  """
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)
  return input_image, real_image



# fonction load_dataset

"""
_Summary_ : fonction qui charge les datasets train et val et les prépare pour l'entrainement
_Args_ : BATCH_SIZE (taille du batch), BUFFER_SIZE (taille du buffer pour le shuffle)
_Returns_ : les datasets d'entrainement et de validation processés
"""

def load_dataset (BATCH_SIZE=1, BUFFER_SIZE=400):
  train_dataset = tf.data.Dataset.list_files(str(path / 'train/*.jpg'))
  train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.cache()
  train_dataset = train_dataset.shuffle(BUFFER_SIZE)
  train_dataset = train_dataset.batch(BATCH_SIZE)
  train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
  val_dataset = tf.data.Dataset.list_files(str(path / 'val/*.jpg'))  
  val_dataset = val_dataset.map(load_image_val,
                                  num_parallel_calls=tf.data.AUTOTUNE)
  val_dataset = val_dataset.cache()
  val_dataset = val_dataset.shuffle(BUFFER_SIZE)
  val_dataset = val_dataset.batch(BATCH_SIZE)
  val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
  return train_dataset, val_dataset




"""

test unitaire pour tester le dataset

import matplotlib.pyplot as plt
import pathlib


train_dataset=load_dataset(BATCH_SIZE=1, BUFFER_SIZE=400)

inp, re = load(str(path / 'train/100.jpg'))

save_dir = os.path.abspath("./graph_results/")

plt.figure()
plt.imshow( inp /255) # normalise de [-1;1] à [0;1]
plt.savefig(f"/ccc/scratch/cont001/ocre/lebouchers/map")
plt.imshow( re /255)
save_path = os.path.join(save_dir, "re.png")
plt.savefig(save_path)

"""
