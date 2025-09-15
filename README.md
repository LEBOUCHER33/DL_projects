# DL_projects

# Image to Image supervised Machine Learning

## Objective

Feasibility study on integrating a supervised Image-to-Image learning model for visual data prediction in numerical simulation codes to accelerate computation times.


## Description

This project aims to implement and evaluate various supervised learning models using pre-trained models from TensorFlow, Mistral AI, and Stability AI, in order to assess their accuracy and the quality of their simulated prediction images.


## Models

### 1- Pix2Pix 
Pix2Pix is an image-to-image translation model using a conditional Generative Adversarial Network (cGAN).
It learns the mapping from input images and converts them into output images.

### 2- Pixtral12B :
Pixtral-12B is a text-to-image and image-to-text model designed to analyze, describe, and transform images.
It encodes both textual prompts and visual inputs into a shared representation, enabling it to generate coherent image descriptions as well as new images conditioned on prompts.

### 3- Stable_diffusion :
Stable Diffusion is a latent diffusion model that combines a variational autoencoder (VAE) to embed images into a compressed latent space and a CLIP-based text encoder to process textual prompts.
By progressively denoising latent representations, the model reconstructs high-resolution images that align with the semantic meaning of the input text. This process allows efficient generation of detailed, high-quality synthetic images from natural language prompts.


## Badges

link to the originel tutorials:
https://www.tensorflow.org/tutorials/generative/pix2pix?hl=fr



## Installation

### 1- Setting up the Working Environment

1- Download the required libraries (compiled PyPI packages) from the internet into a dedicated folder using the shell script script_repo_update.sh.
2- Transfer this folder to the target computing machine.
3- Create a virtual environment on the machine and install the libraries from the local folder using the shell script script_venv_update.sh.

### 2- Clone the Git Repository
git clone https://gitlab.cesta.dam.intra.cea.fr/sl613344/pix2pix.git
link to the computing machine with sshfs command system
create a copy of the repo on computing machine


## Workflow 

1- Data Preprocessing :
Clean, format, and prepare datasets for training.

2- Model Fine-Tuning :
Apply fine-tuning techniques to optimize model performance.
Compare different hyperparameter configurations using methods such as Ray (for distributed tuning) or LoRA (Low-Rank Adaptation).

3- Training :
Train the selected models with the optimized configurations.
Monitor convergence, stability, and resource usage.

4- Performance Evaluation :
Assess the models on relevant metrics (e.g., accuracy, image quality, coherence).
Compare results across models and fine-tuning strategies.





