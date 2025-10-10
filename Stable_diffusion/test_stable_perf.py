
"""
Script qui teste le modèle stable_diffusion avec UNet entrainé avec LoRA

- loading du modèle pré-entrainé Stable_diffusion v1-5 et application de LoRA à l'UNet
- loading du dataset de test
- génération d'images à partir d'images d'entrée du dataset

"""

# 1- Import des librairies


import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms


# 2- loading du modèle pré-entrainé avec LoRA

model_path = Path(
    "./git/ImageMLProject/Stable_diffusion/SD1-5_local"
)
model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to("cuda")

# ajout de LoRA à l'UNet du modèle
model.unet.load_lora_adapter("./my_lora")
print(model.unet)


# 3- Loading du dataset de test

from Stable_diffusion.stable_main_LoRA import test_dataloader



# 4- génération d'images à partir d'images d'entrée du dataset

import os
# on définit output_dir
save_dir = "./test_lora"
os.makedirs(save_dir, exist_ok=True)

# on récupère un batch d'images
batch = next(iter(test_dataloader))
print(batch["pixel_values"].shape) #torch.Size([8, 3, 512, 512])

 
sample_image = batch["pixel_values"][10] # torch.Size([3, 512, 512])

# on convertit ce tensor en PIL image
to_pil = transforms.ToPILImage()
sample_image_pil = to_pil(sample_image).convert("RGB")
sample_image_pil.save(os.path.join(save_dir, "input_image.png"))

# génération de l'image

prediction = model(
    prompt="fluid simulation",
    image=sample_image_pil,
    strength=0.3, # entre 0 et 1, 1 = on se base uniquement sur le prompt
    guidance_scale=3, # plus c'est grand plus on suit le prompt
    num_inference_steps=50, # nombre d'étapes de raffinements
).images[0]

# sauvegarde de l'image
prediction.save(os.path.join(save_dir, "generated_image.png"))

# 5- visualisation des résultats

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(sample_image_pil)
ax[0].set_title("Input Image")
ax[0].axis("off")
ax[1].imshow(prediction)
ax[1].set_title("Generated Image")
ax[1].axis("off")
plt.show()
plt.savefig(os.path.join(save_dir, "comparison.png"))
