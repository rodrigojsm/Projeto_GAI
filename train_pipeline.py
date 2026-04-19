from __future__ import annotations

import sys
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
import csv


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Relative paths (run this notebook from student_start_pack/)
PROJECT_ROOT = Path("")
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
KAGGLE_ROOT = PROJECT_ROOT / 'archive'

if not KAGGLE_ROOT.exists() or not (SCRIPTS_DIR / 'artbench_local_dataset.py').exists():
    raise FileNotFoundError(
        'Could not resolve project folders from relative paths. '
        'Run this notebook from student_start_pack/ or adjust PROJECT_ROOT.'
    )

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if __name__ == '__main__':
    print('PROJECT_ROOT =', PROJECT_ROOT)
    print('KAGGLE_ROOT  =', KAGGLE_ROOT)
else:
    print("worker Starting...")

# Uses your existing project helper to load ArtBench-10 from local Kaggle-style files
from scripts.artbench_local_dataset import load_kaggle_artbench10_splits

hf_ds = load_kaggle_artbench10_splits(KAGGLE_ROOT)
train_hf = hf_ds["train"]
if __name__ == '__main__':
    print("Train size:", len(train_hf))
    print("Columns   :", train_hf.column_names)

label_feature = train_hf.features["label"]
class_names = list(label_feature.names)
num_classes = len(class_names)
if __name__ == '__main__':    
    print("Num classes:", num_classes)
    print("Class names:", class_names)

# Class distribution summary
train_counts = Counter(train_hf["label"])

if __name__ == '__main__':
    print("\nTrain class distribution:")
    for cid, name in enumerate(class_names):
        print(f"  {cid:2d} | {name:>15s} | {train_counts.get(cid, 0):6d}")

# ==========================================
# 1. CONFIGURATION & TOGGLE
# ==========================================
# Change this to True to run the 20% CSV subset, or False for the full/fractional dataset.
USE_CSV_SUBSET = False 

IMAGE_SIZE = 32
BATCH_SIZE = 256
NUM_WORKERS = 4
SEED = 42

# Settings for full/fractional run (Used if USE_CSV_SUBSET = False)
TRAIN_FRACTION = 1.0  

# Settings for CSV run (Used if USE_CSV_SUBSET = True)
TRAINING_CSV_PATH = Path('training_20_percent.csv')
INDEX_COLUMN = 'train_id_original' 


# ==========================================
# 2. SHARED CLASSES & FUNCTIONS
# ==========================================
def safe_num_workers(requested: int) -> int:
    if "ipykernel" in sys.modules and int(requested) > 0:
        print("Notebook kernel detected: forcing num_workers=0 for DataLoader stability.")
        return 0
    return int(requested)

EFFECTIVE_NUM_WORKERS = safe_num_workers(NUM_WORKERS)

transform = T.Compose([
    T.Resize(IMAGE_SIZE, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),  
])

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y, real_idx

def make_subset_indices(n_total: int, fraction: float, seed: int = 42) -> list[int]:
    n_keep = max(1, int(round(n_total * fraction)))
    g = np.random.RandomState(seed)
    idx = np.arange(n_total)
    g.shuffle(idx)
    return idx[:n_keep].tolist()

def load_ids_from_training_csv(csv_path: Path, index_column: str) -> list[int]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    
    ids = []
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        r = csv.DictReader(f)
        if index_column not in (r.fieldnames or []):
            raise ValueError(f"Column {index_column!r} missing. Available: {r.fieldnames}")
        
        for row in r:
            v = str(row.get(index_column, "")).strip()
            if v: ids.append(int(v))
            
    if not ids:
        raise ValueError(f"No ids found in {csv_path}")
    return ids


# ==========================================
# 3. SET SELECTION 
# ==========================================
if __name__ == '__main__':
    import time
    start = time.time()
    
    if USE_CSV_SUBSET:
        print(f"Mode: CSV Subset -> Loading indices from {TRAINING_CSV_PATH}")
        train_indices = load_ids_from_training_csv(TRAINING_CSV_PATH, INDEX_COLUMN)
    else:
        print(f"Mode: Random Subset -> Generating fraction: {TRAIN_FRACTION}")
        train_indices = make_subset_indices(len(train_hf), TRAIN_FRACTION, seed=SEED)


# ==========================================
# 4. DATASET & DATALOADER INITIALIZATION
# ==========================================
    train_ds = HFDatasetTorch(train_hf, transform=transform, indices=train_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=EFFECTIVE_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers= EFFECTIVE_NUM_WORKERS > 0
    )

    print("-" * 30)
    ds_length = len(train_ds)
    print("Train dataset length :", ds_length)
    print("Train batches        :",)


    # ==========================================
    # 5. Training
    # ==========================================

    import matplotlib.pyplot as plt

    def generate_new_images(model, num_images=10, latent_dim=128, device='cpu'):
        #  ALWAYS put the model in evaluation mode before generating
        model.eval()
        
        
        # Tell PyTorch we don't need to track gradients (saves memory & runs faster)
        with torch.no_grad():
            # Create the random noise! 
            z = torch.randn(num_images, latent_dim).to(device)
            
            # Pass the noise through the decode method
            fake_images = model(z)
            
            # Move images back to CPU for plotting
            fake_images = fake_images.cpu()

        # Plot the results
        fig = plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)

            img_tensor = fake_images[i]
            
            # Rearrange dimensions for Matplotlib (Shape [H, W, 3])
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = (img_np + 1.0) / 2.0

            plt.imshow(img_np.clip(0, 1), interpolation='none') 
            plt.axis('off')
            
        plt.tight_layout()
        plt.show() 


    # Setup device, model, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    import sys
    
    import Models.VAE as VAE
    import Models.DCGAN as DCGAN
    import Models.Diffusion as Diffusion

    model = None
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "vae":
            lr = 3e-3
            beta = 0.05,
            model = VAE.Module(in_channels=3, latent_dim=128)
            model.startTraining( 
                train_loader=train_loader, 
                ds_length=ds_length, 
                learning_rate=lr, 
                device=device, 
                beta = beta,
                epochs=30
            )
        elif sys.argv[1].lower() == "dcgan" or sys.argv[1].lower() == 'gan':
            learning_rate = 2e-04
            model = DCGAN.Module(in_channels=3, latent_dim=128)
            model.startTraining( 
                train_loader=train_loader, 
                ds_length=ds_length, 
                learning_rate=lr, 
                device=device, 
                epochs=30
            )
            model.startTraining( 
                train_loader=train_loader, 
                lr=lr, 
                epochs=28,
                ds_length = ds_length
            )
        elif sys.argv[1].lower() == "diffusion" or sys.argv[1].lower() == "diff":
            model =  Diffusion.Module(in_channels=3, device=device)
        else:
            print("Run as \"train_pipeline.py <DCGAN|Diffusion|VAE>\" or  \"train_pipeline.py <DCGAN|Diffusion|VAE> <filename>\" ")
    else:
        raise ValueError("Run as \"train_pipeline.py <DCGAN|Diffusion|VAE>\" or  \"train_pipeline.py <DCGAN|Diffusion|VAE> <filename>\" ")
    
    if (len(sys.argv) > 2):
        saved_weights = torch.load(sys.argv[2], map_location=torch.device('cpu'))
        model.load_state_dict(saved_weights)
    
    end = time.time() - start
    print(f"Completed in {end} seconds!")

    # Save the model to a file (the .pth extension is standard for PyTorch)
    # path = f'{sys.argv[1]}.pth'
    # torch.save(model.state_dict(), path)
    # print(f"Model saved as {Path}")

    # learning_rates = [2e-04]
    # betas = [0.3]

    # best_loss = float('inf') 
    # best_params = {'lr': None, 'beta': None}
    # best_model = None

    # print("Starting Hyperparameter Grid Search...")
    # total = 3
    # i = 0
    # # 2. Loop through every combination
    # for lr in learning_rates:
    #     for b in betas:
    #         print(f"\n--- Testing LR: {lr} | Beta: {b} ---")
            
    #         model = Diffusion.Module(in_channels=3, device=device)
            
    #         avg_loss = model.startTraining( 
    #             train_loader=train_loader, 
    #             lr=lr, 
    #             epochs=100,
    #             ds_length = ds_length
    #         )
    #         end = time.time() - start
    #         print(f"Completed in {end} seconds!")
    #         # 3. Check if this combination is the new champion
    #         if avg_loss < best_loss:
    #             best_loss = avg_loss
    #             best_params['lr'] = lr
    #             best_params['beta'] = b
    #             best_model = model
    #             best_model.generate_new_images(num_images=10, latent_dim=128, device=device)
    #             print(f"🌟 New Best Found! Loss: {best_loss:.4f}")
    #         i+=1
    #         print(f"{i} out of {total} combinations completed")

    # print("\n=========================================")
    # print(f"🏆 Grid Search Complete!")
    # print(f"Best Learning Rate: {best_params['lr']:.2e}")
    # print(f"Best Beta: {best_params['beta']}")
    # print(f"Lowest Loss: {best_loss:.4f}")
    # print("=========================================")

    
    