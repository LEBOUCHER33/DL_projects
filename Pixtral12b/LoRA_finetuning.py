"""
Script de *Fine_tuning* Pixtral12B pour la classification d'images satellite
    -chargement du modèle pré-entraîné Pixtral12B
    -préparation du dataset d'images
    -ajout de LoRA pour le fine-tuning
    -entrainement sur le dataset
    -sauvegarde de LoRA
    -chargement avec le modèle
    -utilisation en local

"""

# 1- import des librairies

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from transformers import AutoProcessor, Trainer, TrainingArguments, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import os
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader



# 1- Loading de Pixtral-12B model et de son processor

model_dir= os.path.abspath("./git/ImageMLProject/Pixtral12B/Pixtral12B_local")
# model
model = LlavaForConditionalGeneration.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for reduced memory usage
    device_map="auto"
)
# processor
processor = AutoProcessor.from_pretrained(model_dir)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token
# nécessaire de définir un pad_token pour indiquer comment compléter les séquences de tokens si trop courtes
tokenizer = AutoTokenizer.from_pretrained(model_dir)




# 2- Dataset : préparation du jeu de données multimodal

## 2-1 Loading du dataset

dataset_path = os.path.abspath('./git/ImageMLProject/Datasets/')
data_dir = Path() / dataset_path / "AID"

data = []

# extraction des paires (image, label)
for d in data_dir.iterdir():
    if not d.is_dir():
        continue
    data.extend([{"label": d.name, "img_path": p} for p in d.iterdir()])

df = pd.DataFrame(data)
classes = [*df["label"].unique()]

"""
On a des objets PosixPath dans le dataset, il faut les convertir en str avant de les formater
"""

df['img_path'] = df['img_path'].astype(str)


## 2-2 formatage du dataset

"""
Workflow :

1- DF → HuggingFace Dataset
2- Dataset → format messages (user + assistant, avec image + texte)
3- Export JSONL si besoin (Pixtral attend ça en input brut)
4- Chargement du JSONL → apply processor.apply_chat_template
5- Processor encode (input_ids, pixel_values, labels)
6- Map sur tout le dataset → processed_dataset
7- Utilisation dans LoRA fine-tuning

"""
# -------------------
## 2-2-1- on transforme le DF dans un format lu par transfomers de HF
# -------------------


dataset = Dataset.from_pandas(df)

# split train/val/test

splits = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

# Puis on re-split le "train" en train/validation
train_valid = splits["train"].train_test_split(test_size=0.1, seed=42)

# Fusion en DatasetDict final
dataset_dict = {
    "train": train_valid["train"],
    "validation": train_valid["test"],
    "test": splits["test"],
}

# sauvegarde du DatasetDict
dataset_save_dir = "./git/ImageMLProject/Pixtral12B/dataset_dict"
dataset_dict.save_to_disk(dataset_save_dir)

# loading du DatasetDick sauvegardé
dataset_dict = DatasetDict.load_from_disk(dataset_save_dir)

print(dataset_dict['train'][0])
"""
output :
{'label': 'Beach', 'img_path': '/path/git/ImageMLProject/Datasets/AID/Beach/beach_230.jpg'}
"""

# nettoyage du dataset

"""
Pixtral ne supporte pas des inputs vide (text ou image)
"""


# Fonction pour filtrer les exemples invalides

def filter_empty(example):
    return example["img_path"] is not None and example["label"] is not None and example["img_path"] != "" and example["label"] != ""



for split in ['train', 'validation', 'test']:
    dataset_dict[split] = dataset_dict[split].filter(filter_empty)

# -------------------
## 2-2-2 Construire le format "messages"
# -------------------

"""
To format the dataset, all vision finetuning tasks should be formatted as follows:

{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Quel est le type de cette scène ?"},
        {"type": "image", "image": "/.../Beach/beach_230.jpg"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "Beach"}
      ]
    }
  ]
}

"""



def format_for_pixtral(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": 'Quel est le label de cette image ?'},
                    {"type": "image", "image": example["img_path"]}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": example["label"]}
                ]
            }
        ]
    }

dataset_dict = dataset_dict.map(format_for_pixtral, remove_columns=dataset_dict["train"].column_names)

print(dataset_dict["train"][0])
"""
output :
{'messages': [{'content': [{'image': None, 'text': 'Quel est le label de cette image ?', 'type': 'text'}, {'image': '/ccc/scratch/cont001/ocre/lebouchers/git/ImageMLProject/Datasets/AID/Beach/beach_230.jpg', 'text': None, 'type': 'image'}], 'role': 'user'}, {'content': [{'image': None, 'text': 'Beach', 'type': 'text'}], 'role': 'assistant'}]}

"""

# -------------------
## 2-2-3 Processing et encoding des inputs : processor
# -------------------

"""
Pixtral ne lit pas directement les objets "messages" → il faut les convertir en prompt + image.
on a un output type :
{'messages': [{'content': [{'image': None, 'text': 'Viaduct', 'type': 'text'}, {'image': '/ccc/scratch/cont001/ocre/lebouchers/git/ImageMLProject/Datasets/AID/Viaduct/viaduct_215.jpg', 'text': None, 'type': 'image'}], 'role': 'user'}, {'content': [{'image': None, 'text': 'Réponse à Viaduct', 'type': 'text'}], 'role': 'assistant'}]}

on veut des entrées type :
{
  "input_ids": ...,       # encodage du texte
  "attention_mask": ...,
  "pixel_values": ... ,   # pour les images
  "labels": ...           # ce que le modèle doit prédire
}

"""

# le modèle prédit les tokens contenus dans labels à partir de input_ids
# il faut masquer le user labels pour que seuls les tokens de l'assistant soit utilisés pour 
# calculer la loss et mettre à jour les poids

def preprocess(example):
    messages = example["messages"]
    # ---------------------------
    # 1. Construire le prompt texte
    # ---------------------------
    prompt = processor.apply_chat_template(messages, tokenize=False)
    # ---------------------------
    # 2. Charger l'image (première image côté user)
    # ---------------------------
    image_path = next(
        (item["image"] for item in messages[0]["content"]
         if item["type"] == "image" and item["image"] is not None),
        None
    )
    image = Image.open(image_path).convert("RGB")
    # ---------------------------
    # 3. Encoder texte + image
    # ---------------------------
    encoded = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding="max_length",
        truncation=False  # important pour éviter mismatch texte/image
    )
    # ---------------------------
    # 4. Masquer le prompt user dans labels
    # ---------------------------
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded["attention_mask"].squeeze(0)
    labels = input_ids.clone()

    # on trouve où commence la réponse (assistant)
    # le processor crée un token spécial "assistant" ou "<assistant>" dans le texte
    # pour simplifier ici, on masque tout avant le premier token de l'assistant
    text = processor.tokenizer.decode(input_ids)
    split_marker = "assistant:"  # selon template utilisé par Pixtral
    if split_marker in text:
        mask_index = text.index(split_marker)
        # on convertit le caractère en token index
        tokens_before = processor.tokenizer(text[:mask_index], add_special_tokens=False)["input_ids"]
        labels[:len(tokens_before)] = -100  # masquer la question/user
    else:
        # si pas trouvé, on ne masque rien
        pass

    # ---------------------------
    # 5. Retourner les features finales
    # ---------------------------
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": encoded["pixel_values"].squeeze(0),
        "labels": labels
    }


processed_dataset = dataset_dict.map(preprocess, remove_columns=["messages"])
























"""
Pixtral et LoRA attendent des fichiers JSON Lines avec un objet par ligne
"""

dataset_dict = DatasetDict(dataset_dict)
print(dataset_dict)

# Sauvegarder en JSONL (une ligne = un exemple)
dataset_dict["train"].to_json("train.jsonl", orient="records", lines=True)
dataset_dict["validation"].to_json("val.jsonl", orient="records", lines=True)
dataset_dict["test"].to_json("test.jsonl", orient="records", lines=True)




# 3- Paramètres de configuration LoRA

## définition des paramètres loRA
lora_config = LoraConfig(
    r=16,  # Rank - Higher values for larger datasets
    lora_alpha=32,  # Scaling factor for LoRA
    use_rslora=True,  # Use RS-LoRA for better performance
    target_modules="all-linear",  # Apply LoRA to all linear layers
    lora_dropout=0.05,  # Dropout rate for LoRA
    bias="none",  # No bias adjustment
    task_type="CAUSAL_LM"  # Task type: Causal Language Modeling
)

## implémentation du modèle + LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 




# 4- Paramètres de training

## 4-1 définir les params de configuration du training : centralise le process
training_args = TrainingArguments(
    output_dir="./pixtral_lora_results",
    eval_steps=250,
    logging_steps=150,
    gradient_accumulation_steps=3,
    save_steps=400,
    per_device_train_batch_size=1,  # Adjust based on hardware capabilities
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    fp16=True,  # Use mixed precision if supported
    report_to="none",
    log_level="info",
    learning_rate=2e-5,
    max_grad_norm=2,
    lr_scheduler_type='linear',
    remove_unused_columns=False
)


## 4-2 Loader le dataset

# pour le trainer il ne faut aucun text (user ou assistant) vide ou image manquante


dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "val.jsonl"
})

print(dataset['train'][0])


# on applique le formatage du dataset pour obtenir la clé "messages"

dataset = dataset.map(
    format_for_pixtral,
    remove_columns=dataset["train"].column_names  # supprime toutes les colonnes existantes
)


print(dataset["train"][0])


## 4-3 Préprocessing du dataset pour le modèle



class PixtralDataCollator:
    def __init__(self, processor):
        self.processor = processor
    def __call__(self, batch):
        texts = []
        images = []
        labels = []
        for ex in batch:
            user_text = ex["messages"][0]["content"][0]["text"]
            assert user_text is not None and len(user_text.strip()) > 0
            img_path = ex["messages"][0]["content"][1]["image"]
            assistant_text = ex["messages"][1]["content"][0]["text"]
            # skip si texte vide ou image manquante
            if not user_text or not img_path:
                continue
    # texte utilisateur (description image)
            texts.append(f"Classify_image : {user_text}")
    # image associée
            images.append(Image.open(img_path).convert("RGB"))
    # labels
            labels.append(assistant_text)
    # encoding des inputs user_text + image
        inputs_encoded = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,   # les pytorch tensors doivent être de même taille : padding complète les trop courts
            truncation=True # tronque les trop longs
        )
    # encoding des labels (= text_assistant)
        labels_enc = self.processor.tokenizer(
            labels,
            return_tensors="pt",
            padding=True,
            truncation=True
    )
        inputs_encoded["labels"] = labels_enc.input_ids # version tokenisée du text_assistant
        return inputs_encoded


data_collator = PixtralDataCollator(processor)
print(callable(data_collator))

"""
processed_dataset = dataset.map(
    preprocess_pixtral,
    remove_columns=dataset["train"].column_names,
    batched=False
)

processed_path = "./git/ImageMLProject/Pixtral12B/processed_dataset"
processed_dataset.save_to_disk(processed_path)
"""


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator = data_collator,
)

# Fine-tuning du modèle
trainer.train()


# Save the fine-tuned model and tokenize
save_dir = os.path.abspath("./git/ImageMLProject/Pixtral12B/")
save_path = Path (save_dir) / "my_lora_pixtral_model"
trainer.save_model(save_path)
processor.tokenizer.save_pretrained(save_path)


