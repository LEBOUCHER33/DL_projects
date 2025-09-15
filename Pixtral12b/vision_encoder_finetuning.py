"""
Script de fine-tuning du modèle Pixtral12B de Mistral AI
    -chargement du modèle pré-entraîné Pixtral12B
    -préparation du dataset d'images
    -extraction du vision_encoder
"""


# 1- import des librairies

import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
from torchinfo import summary

# accélération des téléchargements
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"



# 2- Chargement du modèle

model_id = "mistral-community/pixtral-12b"

# il faut trouver la bonne classe de transformer pour charger le modèle suivant les modifications qu'on veut ajouter
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    trust_remote_code=True, # permet d'accéder au code
    torch_dtype=torch.float16, # plus léger que float32 sans perte de précision
    device_map="auto" # gère les devices
    )

# résumé du modèle
summary(model)



# 3- PixtralVisionModel / Vision_encoder de Pixtral


# Vision encoder uniquement
vision_encoder = model.vision_tower

"""
Objectif :
1- récupérer le pytorch_tensor en sortie du vision_encoder 
2- ajouter un décodeur pour générer des images à partir de ce tenseur
"""


# 4- Préparation du dataset

# 4-1 Option 1 = Dataset avec processor

# chargement du processor de données adapté au modèle
# le processor va encoder les entrées suivant leur type (texte, image, audio) pour avoir la bonne config pour être utilisable par le modèle
processor = AutoProcessor.from_pretrained(model_id)

# data : fichier numpy avec des images 128x128 en niveaux de gris
data = np.load('../Datasets/film_cfd_128_1.npy')
print (data.shape, type(data), data.dtype)
# On a 990 images de taille 128x128 en niveaux de gris : pour Pixtral il faut convertir en format image (uint8) et en RGB

data_uint8 = data.astype(np.uint8)

# On va formater nos images pour qu'elles soient utilisables par l'encoder Pixtral

liste_images = []
for img in data_uint8:
    image = Image.fromarray(img, mode="L").convert("RGB")
    liste_images.append(image)

# L'encoder de Pixtral, PixtralVisionModel, attend en entrée des tenseurs d'images RGB normalisés :
# (pixel_values): torch.FloatTensor# 
#     shape = [batch_size, 3, height, width]

inputs = processor(images=liste_images, text=[""] * len(liste_images), return_tensors="pt")
# attention à supprimer la partie texte, processor attend un argument text

pixel_values = inputs["pixel_values"]
print(pixel_values.shape, pixel_values.dtype)

# on convertit en float16 comme notre modèle
pixel_values = pixel_values.half()
print(type(pixel_values), pixel_values.shape, pixel_values.dtype)

# On récupère nos pytorch_tensors en sortie du Vision_encoder
with torch.no_grad(): # supp le calcul du gradient 
    outputs = vision_encoder(pixel_values)

outputs_tensor = outputs.last_hidden_state
# Next step -> ajouter un decoder pour traduire en image







# 4-2 Option 2 = Dataset sans processor

# On a un array de shape (batch_size, H, W, canal=1), on veut un pt(batch_size, canal=3, H, W)

# on convertit en RGB
data_rgb = np.stack(3*[data], axis=3)
print(data_rgb.shape, data_rgb.dtype)

# Visualisation d'une image
image = Image.fromarray((data_rgb[2]).astype("uint8"))
image

# on convertir les images en float32 et en tensor pytorch
pix_values = torch.from_numpy(data_rgb).float()
# on normalise
pix_values = pix_values / 255.0
# on permute les dimensions pour coller au format VisionEncoderPixtral (B, C, H, W)
pix_values = pix_values.permute(0, 3, 1, 2)
print(pix_values.shape, type(pix_values), pix_values.dtype)

# Pour Pixtral, il est important de centrer/réduire les données afin qu'elles soient plus proches de celles utilisées pour entrainer le modèle (données CLIP-like). Cela va aider à mieux apprendre et généraliser

mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
std  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
pix_values = (pix_values - mean) / std

# on passe en float16 pour coller au modèle
pix_values = pix_values.half()

with torch.no_grad():
    outputs_vis = vision_encoder(pix_values)

# tensor
outputs_vis.last_hidden_state

# SI jamais, on peut le remettre en array
np_array = outputs_vis.last_hidden_state.cpu().float().numpy()
np_array



