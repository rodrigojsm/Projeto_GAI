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
USE_CSV_SUBSET = True 

IMAGE_SIZE = 32
BATCH_SIZE = 64
NUM_WORKERS = 2
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
    )

    print("-" * 30)
    print("Train dataset length :", len(train_ds))
    print("Train batches        :", len(train_loader))


    # ==========================================
    # 5. Training
    # ==========================================

    # Setup device, model, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    import Models.VAE
    import Models.DCGAN
    import Models.Diffusion


    model = VAE.Module(in_channels=3, latent_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 10

    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images, labels, indices = batch 
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed_images, mu, logvar = model(images)
            
            loss = model.loss_function(reconstructed_images, images, mu, logvar)
            
            # Backward pass
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            # Print progress every 50 batches
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} [{batch_idx * len(images)}/{len(train_ds)}] Loss: {loss.item() / len(images):.4f}")

        # Average loss for the epoch
        avg_loss = train_loss / len(train_ds)
        print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")
    
    print("Training Complete!")