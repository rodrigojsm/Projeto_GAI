import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math
from torchvision.utils import save_image

# ==========================================
# 1. Time Embedding & U-Net Modules
# ==========================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Optimized Time Embeddings (Precomputed)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        
        # Precompute and register as a buffer so it automatically moves to the GPU 
        # without re-allocating memory every forward pass.
        self.register_buffer('freqs', torch.exp(torch.arange(half_dim) * -embeddings))

    def forward(self, time):
        embeddings = time[:, None] * self.freqs[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    A simplified U-Net for 32x32 images that takes both an image and a timestep embedding.
    """
    def __init__(self, c_in=3, c_out=3, time_dim=256):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )
        
        # Downsampling path
        self.inc = DoubleConv(c_in, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Time embedding projections for each resolution level
        self.time_emb1 = nn.Linear(time_dim, 128)
        self.time_emb2 = nn.Linear(time_dim, 256)
        
        # Upsampling path
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up_conv1 = DoubleConv(256, 128) # 256 because of skip connection concatenation
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.up_conv2 = DoubleConv(128, 64)  # 128 because of skip connection concatenation
        
        # Output layer
        self.outc = nn.Conv2d(64, c_out, 1)

    def forward(self, x, t):
        # 1. Calculate time embeddings
        t_emb = self.time_mlp(t)
        
        # 2. Downsample
        x1 = self.inc(x)
        x2 = self.down1(x1)
        # Inject time embedding by adding it to the feature maps
        x2 = x2 + self.time_emb1(t_emb).unsqueeze(-1).unsqueeze(-1) 
        
        x3 = self.down2(x2)
        x3 = x3 + self.time_emb2(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # 3. Upsample and concatenate with skip connections
        x = self.up1(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up_conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up_conv2(x)
        
        # 4. Final output predicting the noise
        output = self.outc(x)
        return output