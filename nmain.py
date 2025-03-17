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
    "placeholder_token": "fridgem",  # The token you'll use to trigger your concept
    "initializer_token": "fridge",  # A word that describes your concept
    "learning_rate": 5e-4,
    "scale_lr": True,  
    "max_train_steps": 2000,
    "save_steps": 250,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,
    "mixed_precision": "fp16",
    "seed": 42,
    "concept_folder": "fridgem",  # 3-5 images of your concept
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
   
    # Resize token embeddings and initialize our new placeholder token to the initializer token's embedding
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]
   
    return tokenizer, text_encoder, vae, unet, placeholder_token_id

def freeze_models(text_encoder, vae, unet, placeholder_token_id):
    """
    Freeze all parameters except for the single row in the embedding table
    corresponding to our new placeholder token. We do this by:
      1) Freezing VAE & U-Net completely.
      2) Freezing all text-encoder parameters except embeddings.
      3) Registering a grad hook to zero out gradients for *all* embedding rows
         except the placeholder_token_id row.
    """
    def freeze_params(params):
        for param in params:
            param.requires_grad = False

    # Freeze VAE & U-Net
    freeze_params(vae.parameters())
    freeze_params(unet.parameters())

    # Freeze most of the text encoder except token embeddings
    # (encoder, final layer norm, position embeddings, etc.)
    params_to_freeze = itertools.chain(
        text_encoder.text_model.encoder.parameters(),
        text_encoder.text_model.final_layer_norm.parameters(),
        text_encoder.text_model.embeddings.position_embedding.parameters(),
    )
    freeze_params(params_to_freeze)

    # Turn on grad for the entire embedding matrix...
    embedding = text_encoder.get_input_embeddings()
    embedding.weight.requires_grad_(True)

    # ...then zero out gradient updates everywhere except the placeholder row
    mask = torch.zeros_like(embedding.weight).to(device,dtype=torch.float16)
    mask[placeholder_token_id] = 1.0

    def grads_masker(grad):
        return grad * mask

    embedding.weight.register_hook(grads_masker)

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
        flip_prob=0.5,         # 50% chance of flipping
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

def save_progress(text_encoder, placeholder_token_id, accelerator, save_path):
    """Helper function to save the trained embeddings."""
    logger = get_logger(__name__)
    logger.info(f"Saving embeddings to {save_path}")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {CONFIG["placeholder_token"]: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)

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
 
    # We only pass the embedding weights to the optimizer
    optimizer = torch.optim.AdamW(
        [text_encoder.get_input_embeddings().weight],  # single param with grad hook
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
    # Keep unet in train mode for gradient checkpointing to work
    unet.train()
 
    # Initialize noise scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(CONFIG["pretrained_model"], subfolder="scheduler")
 
    # Recalculate total training steps
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
 
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
 
    pbar = tqdm(range(1, num_train_epochs + 1), desc='Training')
 
    upper_limit = 1000
    max_norm = 1
    loss_history = []
    step = 0

    for epoch in pbar:
        epoch_loss = 0
       
        for item in train_dataloader:
            pixel_values = item["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = item["input_ids"].to(device)
 
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215  # VAE scaling factor

            t = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (pixel_values.shape[0],), device=device
            ).long()

            noise = torch.randn_like(latents)
            noised_latents = noise_scheduler.add_noise(latents, noise, t).to(device, dtype=weight_dtype)

            text_embeddings = text_encoder(input_ids)
            pred_noise = unet(
                sample=noised_latents,
                timestep=t,
                encoder_hidden_states=text_embeddings.last_hidden_state.to(device,dtype=torch.float16)
            ).sample

            if noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, t)
            else:
                target = noise

            loss = F.mse_loss(pred_noise, target)
            loss_history.append(loss.item())
 
            optimizer.zero_grad()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                text_encoder.get_input_embeddings().weight, max_norm=max_norm
            )
            optimizer.step()
           
            if step % CONFIG["save_steps"] == 0 and step > 0:
                save_progress(text_encoder, placeholder_token_id, accelerator, save_path=f"./checkpoints/{step}.pth")

            step += 1
            epoch_loss += loss.item()

            progress_bar.update(1)
            global_step += 1
            if global_step >= max_train_steps:
                break

        print(f"Epoch {epoch} - Loss: {epoch_loss:.4f}")
        if global_step >= max_train_steps:
            break

    # Save final loss history
    loss_folder = "loss_history"
    os.makedirs(f"{loss_folder}/", exist_ok=True)
    with open(f"./{loss_folder}/{CONFIG['placeholder_token']}.json", "w") as f:
        json.dump(loss_history, f)
   
    logger.info(f"Training completed. Peak GPU memory usage: {peak_memory:.2f}GB")

def main():
    print(f"Starting textual inversion training...")
    print(f"Using concept images from: {CONFIG['concept_folder']}")
    print(f"Number of concept images: {len(os.listdir(CONFIG['concept_folder']))}")
   
    # Set seed for reproducibility
    set_seed(CONFIG["seed"])
   
    # ---------------
    # SETUP MODELS
    tokenizer, text_encoder, vae, unet, placeholder_token_id = setup_model_and_tokenizer(CONFIG)
   
    # Debug the dataloader (optional)
    # debug_dataloader(tokenizer, CONFIG)
   
    # ---------------
    # FREEZE AND TRAIN
    freeze_models(text_encoder, vae, unet, placeholder_token_id)
    training_function(text_encoder, vae, unet, tokenizer, placeholder_token_id)
   
    # ---------------
    # SAVE FINAL PIPELINE
    pipeline = StableDiffusionPipeline.from_pretrained(
        CONFIG["pretrained_model"],
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=vae,
        unet=unet,
    )
    pipeline.save_pretrained(CONFIG["output_dir"])
    print(f"Training completed. Model saved to {CONFIG['output_dir']}")

    # ---------------
    # INFERENCE / EXAMPLES
    pipeline = StableDiffusionPipeline.from_pretrained(CONFIG["output_dir"]).to(device)
    pipeline.to(device, dtype=torch.float16)
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    concept = CONFIG["placeholder_token"]
    pipeline_out = pipeline(
        f"A high quality photograph of a {concept}, detailed texture, professional lighting",
        num_inference_steps=50,
        guidance_scale=4
    )
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

    os.makedirs(os.path.join(CONFIG["output_dir"], "test_samples"), exist_ok=True)
    for i, prompt in enumerate(test_prompts):
        image = pipeline(
            prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        image.save(os.path.join(CONFIG["output_dir"], "test_samples", f"sample_{i}.png"))
    
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
        num_images = len(concept_images)
        cols = min(4, num_images)
        rows = math.ceil(num_images / cols)
        while len(concept_images) < rows * cols:
            blank = Image.new('RGB', concept_images[0].size, color=(255, 255, 255))
            concept_images.append(blank)
       
        concept_grid = image_grid(concept_images, rows, cols)
        concept_grid_path = os.path.join(CONFIG["output_dir"], "concept_images_grid.png")
        concept_grid.save(concept_grid_path)
        print(f"Concept images grid saved to {concept_grid_path}")
    else:
        print("No valid images found in the concept folder to create a grid.")

if __name__ == "__main__":
    main()
