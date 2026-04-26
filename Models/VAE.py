import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

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
        # Use Tanh because our transforms now output [-1, 1]
        x = torch.tanh(self.dec_conv3(x)) 
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z)
        return reconstructed_x, mu, logvar

    def loss_function(self, reconstructed_x, x, mu, logvar, beta = 0.1):
        reconstruction_loss = (F.mse_loss(reconstructed_x, x, reduction='sum'))/4

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

        comparison = (comparison + 1.0) / 2.0
        comparison = comparison.clamp(0, 1) # Prevents weird artifacting from float rounding
        
        # 4. Ensure the results folder exists (doesn't crash if it already does)
        os.makedirs("results", exist_ok=True)

        # setting nrow=n puts n images per row. 
        filepath = f"results/training_results_{epoch}.png"
        save_image(comparison.cpu(), filepath, nrow=n)

    def startTraining(self, train_loader, ds_length, learning_rate = 3e-3, beta = 0.05, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),epochs = 10):
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

    def generate_new_images(self, num_images=10, latent_dim=128, device=None, return_images=False, **kwargs):
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #  ALWAYS put the self in evaluation mode before generating
        self.eval()
        
        
        # Tell PyTorch we don't need to track gradients (saves memory & runs faster)
        with torch.no_grad():
            # Create the random noise! 
            z = torch.randn(num_images, latent_dim).to(device)
            
            # Pass the noise through the decode method
            fake_images = self.decode(z)
            
            # Move images back to CPU for plotting
            fake_images = fake_images.cpu()

        if return_images:
            return fake_images

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
    