
"""
Script de fine_tuning de Stable_diffusion avec les adaptaters de la méthode LoRA de HuggingFace

Pour fine_tuner Stable_diffusion on utilisera la lib diffusers qui gère la mécanique de training LoRA


1- Import des librairies
2- loading du modèle pré-entrainé Stable_diffusion v1-5 avec le Pipeline de diffusers
3- loading et processing du dataset
4- configuration des paramètres d'entrainement
5- configuration de LoRA
6- boucle d'entrainement

"""



# ////////////////////////
# 1- Import des librairies
# ////////////////////////

import torch
from pathlib import Path
from diffusers import StableDiffusionImg2ImgPipeline
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn.functional as F
from torchvision import transforms
import csv
from sklearn.model_selection import train_test_split
import random
from itertools import product
import copy

DEBUG = False


# check l'accès aux GPUs
torch.cuda.is_available()



# ////////////////////////////////////
# 2- Loading et processing du dataset
# ////////////////////////////////////

"""
Objectif :
1- création des couples inp/tar
2- mélange des couples
3- split du dataset en train/val/test
4- création des tensors PyTorch [C,H,W] normalisés entre [0,1]
5- création des dataloaders nécessaires pour LoRA

"""

DATA_DIR = Path("./git/ImageMLProject/Stable_diffusion/dataset_velocity")

input_dir = os.path.join(DATA_DIR, "input")
target_dir = os.path.join(DATA_DIR, "target")
all_files = sorted(os.listdir(input_dir))

# shuffle des couples
random.seed(42)
random.shuffle(all_files)


# split du dataset en train/val/test 80%/10%/10%
train_files, tmp_files = train_test_split(all_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(tmp_files, test_size=0.5, random_state=42)

print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

output_dir = Path("./LoRA_results")
os.makedirs(output_dir, exist_ok=True)



class VelocityDataset (Dataset):
    def __init__(self, data_dir, file_list=None):
        self.input_dir = os.path.join(data_dir, "input")
        self.target_dir = os.path.join(data_dir, "target")
        self.files = file_list
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fnames = self.files[idx]
        input_path = os.path.join(self.input_dir, fnames)
        target_path = os.path.join(self.target_dir, fnames)
        # chargement des images .png : loader d'images
        input_image = Image.open(input_path).convert("RGB") # PIL.image (512,512) [0,255]
        target_image = Image.open(target_path).convert("RGB") 
        # transformation des images en np.array
        input_arr = np.array(input_image)
        target_arr = np.array(target_image) # np.ndarray (512,512,3) [0,255]
        # transformation en tensors PyTorch [0,1] (format pour LoRA)
        inp_tensor = torch.from_numpy(input_arr) # torch.tensor [512,512,3] [0,255]
        tar_tensor = torch.from_numpy(target_arr)
        inp_tensor = inp_tensor.permute(2, 0, 1).float() / 255.0 # torch.tensor [3,512,512] [0,1]
        tar_tensor = tar_tensor.permute(2, 0, 1).float() / 255.0 
        return inp_tensor, tar_tensor

"""
possiblité d'utiliser torchvision.transforms :

# ToTensor() :
# 1- conversion PIL.Image -> torch.Tensor
# 2- permute les canaux
# 3- normalise les valeurs entre 0 et 1

transform = transforms.Compose([
    transforms.ConvertImageDtype(torch.float32),
    transforms.ToTensor(),  # convertit automatiquement en C,H,W et normalise entre 0–1
])

inp = transform(Image.open(inp_path).convert("RGB"))
tgt = transform(Image.open(tgt_path).convert("RGB"))

"""
BATCH_SIZE = 1

# loading des datasets
train_dataset = VelocityDataset(DATA_DIR, train_files)
val_dataset = VelocityDataset(DATA_DIR, val_files)
test_dataset = VelocityDataset(DATA_DIR, test_files)

if DEBUG :
    inp, tar = train_dataset[0]
    print(inp.shape, tar.shape, inp.dtype, tar.dtype) # torch.Tensor float32 [3,512,512]
    print(np.min(inp.numpy()), np.max(inp.numpy())) # [0,1]


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # torch.Tensor float32 [1,3,512,512]
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if DEBUG:
    inp, tar = next(iter(train_loader))
    print(inp.shape, tar.shape, inp.dtype, tar.dtype, type(inp), type(tar))
    # torch.Tensor float32 [1,3,512,512]
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(inp[0].permute(1, 2, 0).cpu())  # tensor -> HWC
    plt.title("Input t")
    plt.subplot(1, 2, 2)
    plt.imshow(tar[0].permute(1, 2, 0).cpu())
    plt.title("Target t+1")
    plt.show()



# //////////////////////////////////
# 3- loading du modèle pré-entrainé
# //////////////////////////////////

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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = Path(
    "./git/ImageMLProject/Stable_diffusion/SD1-5_local"
)

model = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, 
    dtype=torch.float16).to(DEVICE)



# /////////////////////////////////////////////
# 4- définition de la fonction d'entrainement
# /////////////////////////////////////////////


def train_lora (model, train_loader, val_loader, lora_config,
                epochs=100, lr=1e-5,
                device=DEVICE, patience=5):
    # integration des adaptaters LoRA dans l'UNet
    model.unet = get_peft_model(model.unet, lora_config)
    optimizer = torch.optim.Adam(model.unet.parameters(), lr=lr)
    train_loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    counter = 0
    # boucle d'entrainement
    for epoch in range(epochs):
        train_loss = 0.0
        model.unet.train()
        for inp, tar in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"): # affichage d'une barre de progression
            inp, tar = inp.to(DEVICE), tar.to(DEVICE) # envoie sur GPU 
            # encodage en latents
            with torch.no_grad():
                latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
            # ajout du bruit gaussien
            timestep = torch.randint(
                low=0, 
                high=model.scheduler.num_train_timesteps, 
                size=(latents_input.shape[0],),
                device = device).long()
            noise = torch.randn_like(latents_input)
            noisy_latents_input = model.scheduler.add_noise(latents_input, noise, timestep)
            # encoding du texte // conditonnement par le texte
            prompt = ["scientific visualization of 2D velocity field, next timestep prediction, grayscale simulation"] * BATCH_SIZE
            inputs = model.tokenizer(prompt, padding="max_length", max_length=model.tokenizer.model_max_length, return_tensors="pt")
            encoder_hidden_states = model.text_encoder(inputs.input_ids.to(device))[0]
            # unet forward pass
            pred_noise = model.unet(noisy_latents_input, 
                              timestep, 
                              encoder_hidden_states).sample 
            # calcul de la perte
            loss = F.mse_loss(pred_noise, noise) # MSE error : le modèle apprend à débruiter et non à predire la target
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        # validation
        # on évalue la perte sur le dataset de validation
        model.unet.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tar in val_loader:
                inp, tar = inp.to(device), tar.to(device) # envoie sur GPU 
                # encodage en latents
                latents_input = model.vae.encode(inp).latent_dist.sample() * 0.18215 # tensor [1,4,64,64]
                # ajout du bruit gaussien
                timestep = torch.randint(
                    low=0, 
                    high=model.scheduler.num_train_timesteps, 
                    size=(latents_input.shape[0],),
                    device = device).long()
                noise = torch.randn_like(latents_input)
                noisy_latents_input = model.scheduler.add_noise(latents_input, noise, timestep) 
                noise_pred = model.unet(noisy_latents_input, 
                                  timestep, 
                                  encoder_hidden_states).sample 
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)
        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # sauvegarde du meilleur modèle
            model.unet.save_pretrained(output_dir)
            model.save_pretrained(output_dir)
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        # affichage des pertes
        print(f"Epoch {epoch+1} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}")
    return train_loss_list, val_loss_list, best_val_loss        



# //////////////////////////////////////////////////
# 5- définition de la fonction de Grid Search LoRA
# //////////////////////////////////////////////////



def run_experiment(model, train_loader, val_loader, 
                   save_dir, 
                   patience=5, epochs=100, lr=1e-5, device=DEVICE):

    os.makedirs(save_dir, exist_ok=True)
    # grille des configs LoRA
    r_values = [4, 8, 16]
    lora_alpha_values = [16, 32, 64]
    dropout_values = [0.0, 0.1, 0.2]
    configs =  list(product(r_values, lora_alpha_values, dropout_values))
    
    results = []

    for (r, alpha, dropout) in configs:
        print(f"Training with LoRA config: r={r}, alpha={alpha}, dropout={dropout}")
        exp_name = f"r{r}_alpha{alpha}_dropout{dropout}"
        save_dir = output_dir / exp_name
        os.makedirs(save_dir, exist_ok=True)

        # Configurer LoRA
        lora_config = LoraConfig(
            r=r,  # nbre de dimensions latentes : plus r est grand, plus    
            lora_alpha=alpha,
            target_modules=["to_q", "to_v"],  # couches attention du UNet
            lora_dropout=dropout,   
            bias="none"
        )
        
        # clone du modèle pour chaque config
        model_copy = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_path, 
            dtype=torch.float16).to(device)
        
        # lancement de l'entrainement
        train_loss_list, val_loss_list, best_val_loss = train_lora(model_copy, train_loader, val_loader,
                                                lora_config=lora_config, 
                                                epochs=epochs, lr=lr, 
                                                patience=patience, device=device)
        
        # enregistrement des resultas
        results.append({
            "config": exp_name,
            "best_val_loss": best_val_loss,
            "val_loss_history": val_loss_list,
            "train_loss_history": train_loss_list
        })

        # sauvegarde de la liste des pertes
        with open(os.path.join(save_dir, "val_loss_history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(val_loss_list, 1):
                writer.writerow([i, loss])
        
        with open(os.path.join(save_dir, "train_loss_history.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss"])
            for i, loss in enumerate(train_loss_list, 1):
                writer.writerow([i, loss])

        # sauvegarde de la courbe des pertes
        plt.figure()
        plt.plot(val_loss_list, label="Validation loss", color="red", linewidth=2, linestyle="--")
        plt.plot(train_loss_list, label="Training loss", color="blue", linewidth=2, linestyle="-")
        plt.xlabel("Epoch") 
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.savefig(os.path.join(save_dir, "losses_curve.png"))
        plt.close()

    return results


# //////////////////////////////////////////
# 6- entrainement avec grid search LoRA
# //////////////////////////////////////////


results = run_experiment(model, train_loader, val_loader, output_dir, epochs=200, lr=1e-5, patience=10, device=DEVICE)



# //////////////////////////////////////
# tri et sélection du meilleur modèle
# /////////////////////////////////////

best = sorted(results, key=lambda x: x["best_val_loss"])[0] # tri par meilleure perte de validation
print(f"Best LoRA config: {best['config']} with Val MSE: {best['best_val_loss']:.6f}")


best_lora_dir = output_dir / best['config']

with open(os.path.join(best_lora_dir, "best_lora_config.txt"), "w") as f:
    f.write(str(best_lora_dir))
