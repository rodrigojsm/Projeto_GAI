import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torchvision.utils import save_image

class Module(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, base_channels=64):
        super(Module, self).__init__()
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # maximum channel depth
        self.max_channels = base_channels * 4
        
        # max_channels * 4 height * 4 width
        self.flattened_size = self.max_channels * 4 * 4

        # ENCODER
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        
        # "Armor": in combination with leaky relu helps prevent permanent neuron deactivation 
        self.enc_bn1 = nn.BatchNorm2d(base_channels) 
        
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        
        # Armor
        self.enc_bn2 = nn.BatchNorm2d(base_channels * 2) 
        
        self.enc_conv3 = nn.Conv2d(base_channels * 2, self.max_channels, kernel_size=4, stride=2, padding=1)
        
        # Armor
        self.enc_bn3 = nn.BatchNorm2d(self.max_channels) 
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        # DECODER
        self.dec_fc = nn.Linear(latent_dim, self.flattened_size)
        
        self.dec_conv1 = nn.ConvTranspose2d(self.max_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        # Armor
        self.dec_bn1 = nn.BatchNorm2d(base_channels * 2)
        
        self.dec_conv2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1)
        # Armor
        self.dec_bn2 = nn.BatchNorm2d(base_channels)
        
        self.dec_conv3 = nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        # Leaky ReLU allows a small, non-zero gradient for negative inputs
        x = F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), 0.2)
        x = F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), 0.2)
        x = F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), 0.2)
        x = x.view(x.size(0), -1) 
        
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-30.0, max=20.0)
        mu = self.fc_mu(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.leaky_relu(self.dec_fc(z), 0.2)
        x = x.view(x.size(0), self.max_channels, 4, 4) 
        
        x = F.leaky_relu(self.dec_conv1(x), 0.2)
        x = F.leaky_relu(self.dec_conv2(x), 0.2)
        
        x = torch.sigmoid(self.dec_conv3(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar, beta = 0.1):
        reconstruction_loss = F.l1_loss(reconstructed_x, x, reduction='sum')

        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return reconstruction_loss + (beta*kl_divergence)
    
    def save_comparison_grid(self, real_images, epoch):
        # Grab up to 10 images from the batch
        n = min(real_images.size(0), 10)
        real_subset = real_images[:n]

        # Generate the reconstructed (fake) images
        self.eval() # Switch to eval mode so things like Dropout don't interfere
        with torch.no_grad():
            reconstructed, _, _ = self(real_subset)
        self.train() # Switch back to training mode immediately after

        # combine the 10 real and 10 fake images into a list of 20 images.
        comparison = torch.cat([real_subset, reconstructed])

        # 4. Ensure the results folder exists (doesn't crash if it already does)
        os.makedirs("results", exist_ok=True)

        # setting nrow=n puts n images per row. 
        filepath = f"results/training_results_{epoch}.png"
        save_image(comparison.cpu(), filepath, nrow=n)

    def startTraining(self, train_loader, ds_length, learning_rate = 5e-4, beta = 0.3, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),epochs = 10):
        epochs_to_warmup = 5
        target_beta = beta
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        avg_loss = None
        print("Starting Training Loop...")
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            beta = min(target_beta, target_beta * (epoch / epochs_to_warmup))
            for batch_idx, batch in enumerate(train_loader):
                images, labels, indices = batch 
                images = images.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed_images, mu, logvar = self(images)
                
                loss = self.loss_function(reconstructed_images, images, mu, logvar, beta = beta)
                
                # Backward pass
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
                
                # Print progress every 50 batches
                if batch_idx % 50 == 0 and ds_length != 0:
                    print(f"Epoch {epoch+1}/{epochs} [{batch_idx * len(images)}/{ds_length}] Loss: {loss.item() / len(images):.4f}")

            # Average loss for the epoch
            if (ds_length):
                avg_loss = train_loss / ds_length
            print(f"====> Epoch: {epoch+1} Average loss: {avg_loss:.4f}")

            if (epoch + 1) % 3 == 0:
                self.save_comparison_grid(images, epoch + 1)
        
        print("Training Complete!")
        return avg_loss
    