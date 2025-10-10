
"""
Script de fine_tuning de Stable_diffusion avec les adaptaters de la méthode LoRA de HuggingFace

- loading du modèle pré-entrainé Stable_diffusion v1-5
- loading et processing du dataset CFD (images RGB 512x512)
- fine_tuning avec LoRA

"""


# 1- Import des librairies

import torch
from pathlib import Path
from datasets import load_dataset
import diffusers
import transformers
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision import transforms
from torch.utils.data import DataLoader

# check l'accès aux GPUs
torch.cuda.is_available()


# 2- loading du modèle pré-entrainé

"""
Sur le O :

import torch
import transformers
from diffusers import StableDiffusionImg2ImgPipeline

model_id = "runwayml/stable-diffusion-v1-5"
model = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda"
)
model.save_pretrained("./$CCCSCRATCHDIR/git/ImageMLProject/Stable_diffusion")
"""

model_path = Path(
    "./git/ImageMLProject/Stable_diffusion/SD1-5_local"
)
# si besoin de charger tout le modèle
model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to("cuda")


# 3- Loading et processing du dataset

dataset_path = Path ("./git/ImageMLProject/Stable_diffusion/data")

## 3-1 loading du dataset avec la lib datasets : formatage du dataset en DatasetDict

dataset = load_dataset("imagefolder", data_dir=dataset_path)

"""
DatasetDict({
    train: Dataset({
        features: ['image'],
        num_rows: 1186
    })
    validation: Dataset({
        features: ['image'],
        num_rows: 394
    })
    test: Dataset({
        features: ['image'],
        num_rows: 398
    })
})

"""
train_dataset = dataset["train"]
val_dataset   = dataset["validation"]
test_dataset  = dataset["test"]

# check un exemple
sample = train_dataset[0]
img = sample["image"]
print(type.img) # PIL.Image
img.show()
img.size # (512,512)


## 3-2 Preprocessing des images 

"""
Stable_diffusion attend un format (Batch, Channel, H=512, W=512) avec des valeurs normalisées [-1,1]

"""
# pipeline de preprocessing avec la lib transformers

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),                     # (C,H,W) en [0,1]
    transforms.Normalize([0.5]*3, [0.5]*3)     # normalisation [-1,1]
])

# preprocessing des images

def transform_fn(example):
    """
    _Summary_: fonction qui applique le preprocessor aux images des datasets
    _Args_ : une image
    _Returns_ : une image formatee

    """
    pixel_values = [transform(img) for img in example["image"]]
    return {"pixel_values": pixel_values}


train_dataset = dataset["train"].with_transform(transform_fn)
val_dataset   = dataset["validation"].with_transform(transform_fn)
test_dataset  = dataset["test"].with_transform(transform_fn)


## 3-3 loading du dataset avec la lib DataLoader (nécessaire pour utiliser LoRA)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader   = DataLoader(val_dataset, batch_size=8)
test_dataloader  = DataLoader(test_dataset, batch_size=8)

"""
output :
batch = next(iter(train_dataloader))
print(batch.keys())                  # -> dict_keys(['pixel_values'])
print(batch["pixel_values"].shape)   # torch.Size([8, 3, 512, 512])

"""

# 4- Fine-tuning avec LoRA : on ré-entraine que les poids légers sur notre dataset

from peft import LoraConfig

# définir la config LoRA 
# les paramètres vont définir où et comment on ajoute des pts entrainables

lora_config = LoraConfig(
    r=4, # rang : on ajoute une correction de rang r
    lora_alpha=16, # valeur de scaling (effet LoRA réel = lora_alpha/r), plus c'est grand plus l'effet est grand
    target_modules=["to_q", "to_v"], # liste des couches où appliquer LoRA (q=query, v=value)
    lora_dropout=0.05, # dropout pour régulariser
    bias="none", # pas de biais
    task_type="CAUSAL_LM",
)

# Diffusers fournit déjà un mécanisme pour appliquer LoRA aux sous-modules
from diffusers import UNet2DConditionModel

# loading de l'UNet du modèle Stable_diffusion
unet = UNet2DConditionModel.from_pretrained(
    model_path,
    subfolder="unet",
)

# application de LoRA à l'UNet
unet.add_adapter(lora_config)

# check le nombre de paramètres entrainables
def print_trainable_parameters(model):
    """
    _Summary_: affiche le nombre de paramètres entrainables dans le modèle
    _Args_ : un modèle PyTorch
    _Returns_ : None

    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"Trainable params: {trainable_params} || All params: {all_param} || Trainable%: {100 * trainable_params / all_param}"
    )

print_trainable_parameters(unet)

# 5- Entrainement du modèle avec LoRA

unet.enable_adapters()

# 6- Sauvegarde du modèle LoRA

# sauvegarde des poids LoRA
unet.save_lora_adapter("./my_lora")

# pour recharger le modèle LoRA plus tard
model.unet.load_lora_adapter("./my_lora")