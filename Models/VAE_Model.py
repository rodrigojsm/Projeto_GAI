import torch.nn as nn
import torch.nn.functional as F
import torch

class Module(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super(Module, self).__init__()
        self.latent_dim = latent_dim
        
        # ENCODER: Compresses 32x32 image down to a flattened vector
        self.enc_conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1) # -> 16x16
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)          # -> 8x8
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)         # -> 4x4
        
        # 128 channels * 4 * 4 spatial dimensions = 2048
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)
        
        # DECODER: Expands latent vector back to 32x32 image
        self.dec_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.dec_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # -> 8x8
        self.dec_conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # -> 16x16
        self.dec_conv3 = nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1) # -> 32x32

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1) # Flatten
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc(z))
        x = x.view(x.size(0), 128, 4, 4) # Unflatten
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        # Use Sigmoid on the final layer because our transforms output [0, 1]
        x = torch.sigmoid(self.dec_conv3(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar):
        reconstraction_loss = F.mse_loss(reconstructed_x, x, reduction='sum')

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return reconstraction_loss + kl_divergence