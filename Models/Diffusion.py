import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import math
from torchvision.utils import save_image
from Models.UNET import UNet


# Diffusion Process & Training Class
class Module(nn.Module):
    def __init__(self, timesteps=1000, in_channels=3, img_size=32, device=("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.timesteps = timesteps
        self.img_size = img_size
        self.device = device
        
        # Initialize our U-Net
        self.model = UNet(c_in=in_channels, c_out=in_channels).to(device)
        
        # Define the variance schedule (Beta) - linear schedule from 1e-4 to 0.02
        self.beta = torch.linspace(1e-4, 0.02, timesteps).to(device)
        
        # Alphas define how much of the original image is kept at each step
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion_sample(self, x_0, t):
        """
        Adds noise to the original image x_0 at a specific timestep t.
        This is the forward process: q(x_t | x_0)
        """
        noise = torch.randn_like(x_0).to(self.device)
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None, None]
        
        # Mathematical formula to get the noisy image at step t in one shot
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * noise
        return x_t, noise

    @torch.no_grad()
    def sample(self, n=10, x_T=None):
        """
        Generates new images by starting from pure noise and iteratively denoising.
        This is the reverse process: p(x_{t-1} | x_t)
        """
        self.model.eval()
        
        # If no fixed noise is provided, generate random noise
        if x_T is None:
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        else:
            x = x_T.clone().to(self.device)

        # Iterate backwards from T to 0
        for i in reversed(range(self.timesteps)):
            # Time tensor for the current batch
            t = (torch.ones(n) * i).long().to(self.device)
            
            # Predict the noise currently in the image
            predicted_noise = self.model(x, t)
            
            # Extract scheduling constants for this timestep
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_hat[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            
            # Remove a fraction of the predicted noise
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x) # No noise added at the final step
            
            # The core mathematical update step for DDPMs
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        self.model.train()
        return x

    def save_diffusion_comparison_grid(self, real_images, fixed_noise, epoch):
        """
        Saves a grid of Real Images (top row) vs Fake Images generated from fixed noise (bottom row).
        Adapted from your GAN script.
        """
        # 1. Grab up to 10 real images from the current batch
        n = min(real_images.size(0), 10)
        real_subset = real_images[:n].to(self.device)
        
        # 2. Grab the same number of fixed noise vectors (now 3D image tensors)
        noise_subset = fixed_noise[:n].to(self.device)

        # 3. Generate the FAKE images
        # Diffusion generation is slow! We only need to switch modes inside sample()
        fake_images = self.sample(n=n, x_T=noise_subset)

        # 4. Combine the real and fake images into a list of 20 images.
        comparison = torch.cat([real_subset, fake_images])

        # 5. Ensure the results folder exists
        os.makedirs("results", exist_ok=True)

        # 6. Save the grid
        filepath = f"results/diffusion_training_results_{epoch}.png"
        
        # Dataset and models are in range [-1, 1]. save_image needs [0, 1]
        save_image(comparison.cpu(), filepath, nrow=n, normalize=True, value_range=(-1, 1))

    def generate_new_images(self, num_images=10, device=None, return_images=False, **kwargs):
        """
        Generates and plots n images using your dynamic min-max auto-contrast fix.
        """
        # self.sample() handles the eval() switching 
        
        print(f"Generating {num_images} images... (This might take a minute)")
        fake_images = self.sample(n=num_images).cpu()

        if return_images:
            return fake_images

        fig = plt.figure(figsize=(15, 5))
        for i in range(num_images):
            plt.subplot(2, 5, i + 1)
            img_tensor = fake_images[i]
            img_np = img_tensor.permute(1, 2, 0).numpy()
            
            # Dataset and models are now in range [-1, 1].
            # Map the values to [0, 1] for matplotlib.
            img_np = (img_np + 1) / 2.0

            plt.imshow(img_np.clip(0, 1), interpolation='none') 
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()

    def startTraining(self, train_loader, epochs, ds_length, lr=2e-4):
        # Let PyTorch find the fastest convolution algorithms for your hardware
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        l1_loss = nn.L1Loss()
        
        fixed_noise = torch.randn(10, 3, self.img_size, self.img_size).to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_idx, data in enumerate(train_loader):
                real_images = data[0].to(self.device) if isinstance(data, list) else data.to(self.device)
                batch_size = real_images.shape[0]
                
                t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
                x_t, actual_noise = self.forward_diffusion_sample(real_images, t)
                
                optimizer.zero_grad()
                
                # Notice: The autocast context manager is completely gone. 
                # Just pure, fast FP32 math.
                predicted_noise = self.model(x_t, t)
                loss = l1_loss(predicted_noise, actual_noise)
                
                loss.backward()
                
                # We still keep clipping to keep the model stable!
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_loss += loss.item()
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} [{batch_idx * len(real_images)}/{ds_length}] Loss: {loss.item() *10000 / len(real_images):.4f}")

            avg_loss = epoch_loss*10000 / ds_length
            print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {avg_loss:.4f}")

            # Every 3 epochs, save the comparison grid
            if (epoch + 1) % 3 == 0:
                print(f"Saving grid at epoch {epoch+1}...")
                self.save_diffusion_comparison_grid(real_images, fixed_noise, epoch + 1)

        return avg_loss
