import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import torch
import math
import numpy as np
from torchvision.utils import save_image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

import Models.VAE as VAE
import Models.DCGAN as DCGAN
import Models.Diffusion as Diffusion

from train_pipeline import train_hf, transform, HFDatasetTorch, make_subset_indices, SEED
from torch.utils.data import DataLoader

def map_to_uint8(tensor):
    """Maps [-1, 1] float tensor to [0, 255] uint8 tensor for metrics."""
    # (tensor + 1) / 2 maps to [0, 1]
    tensor_01 = (tensor + 1.0) / 2.0
    tensor_01 = tensor_01.clamp(0, 1)
    return (tensor_01 * 255).to(torch.uint8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = "validation_results"
    os.makedirs(output_dir, exist_ok=True)

    args_lower = [arg.lower() for arg in sys.argv[1:]]
    if "fulltest" in args_lower:
        n_images = 5000
        n_repetitions = 10
        args_lower.remove("fulltest")
        print("--- Running FULL TEST (5000 images, 10 reps) ---")
    else:
        n_images = 1000
        n_repetitions = 3
        print("--- Running SHORT TEST (1000 images, 3 reps) ---")

    batch_size = 50  # Reduced to avoid CUDA Out of Memory
    skip_models = [arg for arg in args_lower if arg in ["vae", "dcgan", "gan", "diffusion", "diff"]]

    # 1. Gather 5000 real images from the dataset
    print(f"Gathering {n_images} real images from train_loader...")
    
    # Initialize train_loader manually
    train_indices = make_subset_indices(len(train_hf), 1.0, seed=SEED)
    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
    
    real_images_list = []
    collected_real = 0
    for batch in train_loader:
        imgs, _, _ = batch
        remaining = n_images - collected_real
        if imgs.size(0) > remaining:
            imgs = imgs[:remaining]
        
        real_images_list.append(imgs)
        collected_real += imgs.size(0)
        if collected_real >= n_images:
            break
            
    real_images = torch.cat(real_images_list, dim=0)
    real_images_uint8 = map_to_uint8(real_images)
    print(f"Collected {real_images_uint8.size(0)} real images. Shape: {real_images_uint8.shape}, dtype: {real_images_uint8.dtype}")

    models = {
        "VAE": {"module": VAE.Module(in_channels=3, latent_dim=128), "weights": "vae.pth"},
        "DCGAN": {"module": DCGAN.Module(in_channels=3, latent_dim=128), "weights": "gan.pth"},
        "Diffusion": {"module": Diffusion.Module(in_channels=3, device=device), "weights": "diff.pth"}
    }

    for model_name, info in models.items():
        # Handle aliases (e.g. diff -> diffusion, gan -> dcgan)
        alias = {"Diffusion": "diff", "DCGAN": "gan"}.get(model_name, model_name.lower())
        if model_name.lower() in skip_models or alias in skip_models:
            print(f"\n{'='*50}\nSkipping {model_name} as requested by arguments...\n{'='*50}")
            continue

        print(f"\n{'='*50}\nEvaluating {model_name}...\n{'='*50}")
        model = info["module"].to(device)
        weights_path = info["weights"]
        
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Successfully loaded weights from {weights_path}.")
        else:
            print(f"Warning: {weights_path} not found! Generating with untrained {model_name}.")

        fid_scores = []
        kid_scores = []

        for rep in range(n_repetitions):
            print(f"\n--- {model_name} | Repetition {rep + 1}/{n_repetitions} ---")
            
            # Change random seed for each repetition to ensure different noise
            torch.manual_seed(42 + rep)
            
            # Initialize metrics for this repetition
            fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
            kid = KernelInceptionDistance(subset_size=100, subsets=50, normalize=False).to(device)
            
            # Feed real images to metrics
            # Batch the real image feeding to avoid OOM in metric feature extractor
            print("  Feeding real images to metrics...")
            with torch.no_grad():
                for i in range(0, n_images, batch_size):
                    real_batch = real_images_uint8[i:i+batch_size].to(device)
                    fid.update(real_batch, real=True)
                    kid.update(real_batch, real=True)
                    print(f"    Fed {min(i + batch_size, n_images)}/{n_images} real images...")

            print(f"  Generating {n_images} fake images...")
            fake_images_list = []
            with torch.no_grad():
                for i in range(0, n_images, batch_size):
                    curr_n = min(batch_size, n_images - i)
                    
                    # Generate fake images
                    imgs = model.generate_new_images(num_images=curr_n, device=device, return_images=True)
                    
                    # Save a grid of 50 images only on the very first repetition
                    if rep == 0 and i == 0:
                        grid_imgs = imgs[:50]
                        filepath = os.path.join(output_dir, f"{model_name}.png")
                        print(f"  -> Saving grid of 50 images to {filepath}...")
                        save_image(grid_imgs, filepath, nrow=10, normalize=True, value_range=(-1, 1))

                    # Map to uint8 and feed to metrics
                    imgs_uint8 = map_to_uint8(imgs).to(device)
                    fid.update(imgs_uint8, real=False)
                    kid.update(imgs_uint8, real=False)
                    
                    print(f"    Generated batch: {min(i + batch_size, n_images)}/{n_images}")

            print("  Computing metrics...")
            with torch.no_grad():
                fid_val = fid.compute().item()
                kid_mean, kid_std = kid.compute()
                kid_val = kid_mean.item()
            
            print(f"  Repetition {rep + 1} Results -> FID: {fid_val:.4f} | KID: {kid_val:.4f}")
            fid_scores.append(fid_val)
            kid_scores.append(kid_val)

        # Final Statistics for this model
        fid_mean = np.mean(fid_scores)
        fid_std = np.std(fid_scores)
        kid_mean_all = np.mean(kid_scores)
        kid_std_all = np.std(kid_scores)
        
        print(f"\n[{model_name}] FINAL STATISTICS (over {n_repetitions} repetitions):")
        print(f"  FID: {fid_mean:.4f} ± {fid_std:.4f}")
        print(f"  KID: {kid_mean_all:.4f} ± {kid_std_all:.4f}")

if __name__ == "__main__":
    main()
