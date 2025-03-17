# main.py
import argparse
import itertools
import math
import os
import random
from typing import Dict, List
import logging
import json

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
    DPMSolverMultistepScheduler
)
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
 
from templates import imagenet_templates_small, imagenet_style_templates_small
from textual_inversion_dataset import TextualInversionDataset
from debug_utils import debug_dataloader
 
# scoring time
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal import CLIPScore
import torchvision.transforms as T


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = get_logger(__name__, log_level="INFO")
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
CONFIG = {
    "pretrained_model": "stabilityai/stable-diffusion-2",
    "what_to_teach": "object",  # Choose between "object" or "style"
    "placeholder_token": "dwayner",  # The token you'll use to trigger your concept
    "initializer_token": "man",  # A word that describes your concept
    "learning_rate": 5e-4,
    "scale_lr": True,  
    "max_train_steps": 2000,
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "concept_folder": "dwayner", # TODO: Change this to your concept folder,  sec 1.1 Concept Preparation
}
# Automatically set output_dir based on concept_folder
CONFIG["output_dir"] = "output_" + CONFIG["concept_folder"].rstrip("/") + "/"
os.makedirs(CONFIG["concept_folder"], exist_ok=True)
os.makedirs(CONFIG["output_dir"], exist_ok=True)
 
if not os.listdir(CONFIG["concept_folder"]):
    raise ValueError(
        f"The concept folder '{CONFIG['concept_folder']}' is empty! "
        "Please add 3-5 images of your concept before running the training."
    )
 
 
def image_grid(imgs: List[Image.Image], rows: int, cols: int) -> Image.Image:
    """Create a grid of images."""
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid
 
def setup_model_and_tokenizer(config: Dict) -> tuple:
    """Setup the model components and tokenizer."""
    tokenizer = CLIPTokenizer.from_pretrained(config["pretrained_model"], subfolder="tokenizer")
   
    # Add placeholder token
    num_added_tokens = tokenizer.add_tokens(config["placeholder_token"])
    if num_added_tokens == 0:
        raise ValueError(f"Token {config['placeholder_token']} already exists!")
       
    # Get token ids
    token_ids = tokenizer.encode(config["initializer_token"], add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("Initializer token must be a single token!")
       
    initializer_token_id = token_ids[0]
    placeholder_token_id = tokenizer.convert_tokens_to_ids(config["placeholder_token"])
   
    # Load models
    text_encoder = CLIPTextModel.from_pretrained(config["pretrained_model"], subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(config["pretrained_model"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["pretrained_model"], subfolder="unet")
   
    # Initialize placeholder token
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
   
    return tokenizer, text_encoder, vae, unet, placeholder_token_id
 
def freeze_models(text_encoder, vae, unet):
    """Freeze all parameters except the token embeddings."""
    def freeze_params(params):
        for param in params:
            param.requires_grad = False
           
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)
 
def create_dataloader(batch_size, tokenizer):
    """Create the training dataloader."""
    train_dataset = TextualInversionDataset(
        data_root=CONFIG["concept_folder"],
        tokenizer=tokenizer,
        size=512,
        placeholder_token=CONFIG["placeholder_token"],
        repeats=100,
        learnable_property=CONFIG["what_to_teach"],
        center_crop_prob=0.5,  # 50% chance of center cropping
        flip_prob=0.5,  # 50% chance of flipping
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
 
def get_gpu_memory_info():
    """Get current and peak GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return 0, 0
    current = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
    peak = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
    return current, peak
 
def training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id):
    train_batch_size = CONFIG["train_batch_size"]
    gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"]
    learning_rate = CONFIG["learning_rate"]
    max_train_steps = CONFIG["max_train_steps"]
    output_dir = CONFIG["output_dir"]
    gradient_checkpointing = CONFIG["gradient_checkpointing"]
 
    # Initialize peak memory tracking
    peak_memory = 0
   
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=CONFIG["mixed_precision"]
    )
 
    if gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()
 
    train_dataloader = create_dataloader(train_batch_size, tokenizer)
    train_dataset = train_dataloader.dataset
 
    if CONFIG["scale_lr"]:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )
 
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=learning_rate,
    )
 
    text_encoder, optimizer, train_dataloader = accelerator.prepare(
        text_encoder, optimizer, train_dataloader
    )
 
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
 
    # Move vae and unet to device
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
 
    # Keep vae in eval mode as we don't train it
    vae.eval()
    # Keep unet in train mode to enable gradient checkpointing
    unet.train()
 
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model"], subfolder="scheduler")
 
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
 
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
 
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
 
    pbar = tqdm(range(1, num_train_epochs + 1), desc='Training')
 
    # t = torch.tensor([500])
    upper_limit = 1000
    max_norm = 1
 
    MSE_loss = torch.nn.MSELoss()
 
    loss_history = []
    step = 0
    for epoch in pbar:
        epoch_loss = 0
       
        for item in train_dataloader:
            pixel_values = item["pixel_values"].to(device,dtype=weight_dtype)
            input_ids = item["input_ids"].to(device)
 
 
            # latents = vae.encode(pixel_values.half()).latent_dist.sample() #try this one as well
            latents = vae.encode(pixel_values.to(device,dtype=weight_dtype)).latent_dist.sample()  #try, mean, mode, sample
            latents = latents * 0.18215 # to avoid all black, scaling is necessary

            # t = torch.randint(low=1, high=upper_limit, size=(pixel_values.shape[0],), device=device)
            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (CONFIG["train_batch_size"],), device=device
            ).long()


            noise = torch.randn_like(latents)
 
            noised_latents = noise_scheduler.add_noise(latents, noise, t).to(device,dtype=weight_dtype)
 
            text_embeddings = text_encoder(input_ids)
 
            # pred_noise=unet(sample = noised_latents, timestep = t, encoder_hidden_states = text_embeddings.last_hidden_state.to(device,dtype=weight_dtype))
            pred_noise=unet(sample = noised_latents, timestep = t, encoder_hidden_states = text_embeddings.last_hidden_state.to(device,dtype=torch.float16))
            # pred_noise = pred_noise.to_tuple()[0]
            pred_noise = pred_noise.sample

            if noise_scheduler.config.prediction_type == "v_prediction":
                # Convert the unet's v to real noise
                target = noise_scheduler.get_velocity(latents, noise, t)
            else:
                # e-pred (v1 style)
                target = noise

            loss = F.mse_loss(pred_noise, target)
            loss_history.append(loss.item())
 
            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), max_norm=max_norm)
            optimizer.step()
            ## Gradient Clipping
            # loss.backward()
            # # torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=max_norm)
            # torch.nn.utils.clip_grad_norm_(text_encoder.get_input_embeddings().parameters(), max_norm=max_norm)
           
            # optimizer.step()
           
            if not (step%CONFIG["save_steps"]):
                save_progress(text_encoder, placeholder_token_id, accelerator, save_path=f"./checkpoints/{step}.pth")
                #torch.save(text_encoder.state_dict(), f"/work3/s214734/02501/ADLCV_Exercise_2_3/checkpoints/checkpoint_{step}.pt")
 
            step += 1
 
            epoch_loss += loss.item()
 
        print(epoch_loss)
 
    loss_folder = "loss_history"
    os.makedirs(f"{loss_folder}/", exist_ok=True)
    with open(f"./{loss_folder}/{CONFIG['placeholder_token']}.json", "w") as f:
        json.dump(loss_history, f)

   ### TODO: Implement the training loop here for Section 1.2 Embedding Training
   ###
   ### You need to:
   ### 1. Loop through epochs and batches
   ### 2. Process images through VAE to get latents
   ### 3. Add noise to latents using the noise scheduler
   ### 4. Get text embeddings from the text encoder
   ### 5. Predict noise with UNet and calculate loss
   ### 6. Update only the embeddings for the placeholder token ##CHECK THIS!
   ### 7. Save checkpoints at specified intervals
   ###
   ### Refer to the main.py file for implementation details
   # ...
   #########################################################
   
    logger.info(f"Training completed. Peak GPU memory usage: {peak_memory:.2f}GB")
 
def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """Helper function to save the trained embeddings."""
    logger = get_logger(__name__)
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {CONFIG["placeholder_token"]: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)
 
def main():
    print(f"Starting textual inversion training...")
    print(f"Using concept images from: {CONFIG['concept_folder']}")
    print(f"Number of concept images: {len(os.listdir(CONFIG['concept_folder']))}")
   
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
   
    ############ Start comment out
    # Setup
    tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
   
    # Debug dataloader before training
    debug_dataloader(tokenizer, CONFIG)
   
    # Continue with training
    freeze_models(text_encoder, vae, unet)
   
    # Train
    training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
   
    # Save the final model
    pipeline = StableDiffusionPipeline.from_pretrained(
        CONFIG["pretrained_model"],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(CONFIG["output_dir"])
    print(f"Training completed. Model saved to {CONFIG['output_dir']}")
 
    #### for inference
    # Make sure to set all models to eval mode
    text_encoder.eval()
    vae.eval()
    unet.eval()

    pipeline = StableDiffusionPipeline.from_pretrained(
        CONFIG["pretrained_model"],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )    
    ############# End comment out
    
    # Load the pipeline with the trained components
    pipeline = StableDiffusionPipeline.from_pretrained(CONFIG["output_dir"]).to(device)

    # Move to device and set precision
    pipeline.to(device, dtype=torch.float16)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # Generate with better prompting and parameters
    concept = CONFIG["placeholder_token"]
    pipeline_out = pipeline(
        f"A high quality photograph of a {concept}, detailed texture, professional lighting",
        num_inference_steps=50,
        guidance_scale=4
    )

    # Save the output
    pipeline_out.images[0].save(os.path.join(CONFIG["output_dir"], f"generated_{concept}.png"))
 
    test_prompts = [
        f"a photo of {concept}",
        f"a high quality photograph of {concept}",
        f"a painting of {concept} with detailed features",
        f"{concept} in a natural setting",
        f"a {concept} on a sand dune",
        f"a {concept} in space with stars and planets night time",
        f"two {concept} next to one another",
        f"a {concept} on a red planet mars looking like a desert",
        f"a {concept} on a grey moon with craters",
        f"a {concept} in a kitchen",
        f"a moon with craters"
    ]

    # print("#", tokenizer.tokenize(f"a {concept}"))
    # Create folder for test images
    os.makedirs(os.path.join(CONFIG["output_dir"], "test_samples"), exist_ok=True)

    # Generate images for each prompt
    for i, prompt in enumerate(test_prompts):
        image = pipeline(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        # Save the image
        image.save(os.path.join(CONFIG["output_dir"], "test_samples", f"sample_{i}.png"))


    # Copy concept folder images as a grid in the output folder
    print("Creating a grid of concept images...")
    concept_images = []
    for image_file in os.listdir(CONFIG["concept_folder"]):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            image_path = os.path.join(CONFIG["concept_folder"], image_file)
            try:
                img = Image.open(image_path).convert('RGB')
                concept_images.append(img)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
   
    if concept_images:
        # Calculate grid dimensions
        num_images = len(concept_images)
        cols = min(4, num_images)  # Max 4 columns
        rows = math.ceil(num_images / cols)
       
        # Pad with blank images if needed
        while len(concept_images) < rows * cols:
            blank = Image.new('RGB', concept_images[0].size, color=(255, 255, 255))
            concept_images.append(blank)
       
        # Create and save the grid
        concept_grid = image_grid(concept_images, rows, cols)
        concept_grid_path = os.path.join(CONFIG["output_dir"], "concept_images_grid.png")
        concept_grid.save(concept_grid_path)
        print(f"Concept images grid saved to {concept_grid_path}")
    else:
        print("No valid images found in the concept folder to create a grid.")
 
    # # Sanity check
    # ids = pipeline.tokenizer("A photo of himiko").input_ids
    # decoded = pipeline.tokenizer.convert_ids_to_tokens(ids[0])
    # print("Tokenized prompt:", decoded, "\nids: ", ids)
   
    # # Setting up pipeline stuff
    # pipeline.to(device, dtype=torch.float16)
    # pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)


    # # show output
    # pipeline_out = pipeline("A photo of himiko")
    # plt.imshow(pipeline_out.images[0])
    # plt.savefig("cool-image.png")
 
 
    # 1.3 Concept Generation
    #  INSTRUCTIONS:
    # TODO: Implement the concept generation section here
    #
    # In this section, you will generate example images using your trained model.
    # This helps evaluate how well your model learned the concept.
    #
    # 1. First, load the trained pipeline from your output directory
    # 2. Configure the DPMSolverMultistepScheduler for efficient sampling
    # 3. Move the model to GPU using the .to("cuda") method
    # 4. Create a test prompt that includes your placeholder token
    # 5. Generate a small batch of images (2 samples is sufficient)
    # 6. Arrange the generated images in a grid for easy viewing
    # 7. Save the grid to your output directory
    #
    # Parameters to experiment with:
    # - num_inference_steps: Higher values (30-50) give better quality but take longer
    # - guidance_scale: Values between 7-9 typically work well
    # - Try different prompts to see how your concept generalizes
    #
    # IMPORTANT: Make sure your GPU has enough memory before running this section!
 
 
if __name__ == "__main__":
    main()