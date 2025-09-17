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

https://mistral.ai/fr/news/unlocking-potential-vision-language-models-satellite-imagery-fine-tuning

https://huggingface.co/radames/stable-diffusion-v1-5-img2img



## Installation

### Setting up the Working Environment

#### 1- local environment

1- Download dependencies : retrieve the required libraries (compiled PyPI packages) from the internet into a dedicated folder 



2- Transfer files : store the downloaded packages in a local folder

3- Create a virtual environment : initialize a Python virtual environment

4- Install dependencies: Install all libraries from the local folder 


#### 2- Remote Supercomputer Environment

1- Transfer packages: Copy the folder containing the compiled PyPI packages to the supercomputer’s storage

2- Module setup : load the compatibility combination GNU(compiler)/CUDA(GPU acceleration)/MPI(distributed or parallel execution)/Python 

3- Resource Management: Using a job submission script to ensure correct allocation of CPUs, GPUs, and memory

4- Set up environment: Create a Python virtual environment

5- Install dependencies: Run the script script_venv.sh on the supercomputer to install libraries from the transferred folder.

6- Job scheduling: Use the supercomputer’s scheduler to submit training and evaluation jobs


### 3- Code Access and Remote Linking

1- Cloning the Repository : retrieve the project source from the remote Git repositiry 

2- Navigate into the cloned project directory to access scripts and configuration files

3- Use sshfs to mount a remote directory from the supercomputer onto the local machine




### 2- Code Access and Remote Linking

1- Cloning the Repository : retrieve the project source from the remote Git repositiry 
2- Navigate into the cloned project directory to access scripts and configuration files
3- Use sshfs to mount a remote directory from the supercomputer onto the local machine


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





