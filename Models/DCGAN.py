import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image
import torch.optim as optim

# counterfeiter
class DCGenerator(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, base_channels=64):
        super(DCGenerator, self).__init__()
        
        # We set bias=False on Convs because BatchNorm mathematically cancels out biases anyway!
        self.net = nn.Sequential(
            # Input: (batch_size, latent_dim, 1, 1)
            # Output: (batch_size, 256, 4, 4)
            nn.ConvTranspose2d(latent_dim, base_channels * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            
            # Input: (batch_size, 256, 4, 4)
            # Output: (batch_size, 128, 8, 8)
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            
            # Input: (batch_size, 128, 8, 8)
            # Output: (batch_size, 64, 16, 16)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            
            # Input: (batch_size, 64, 16, 16)
            # Output: (batch_size, 3, 32, 32)
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            
            # The Tanh activation forces the fake image into the [-1, 1] color space
            nn.Tanh()
        )

    def forward(self, z):
        # The generator expects a 4D tensor (Batch, Channels, Height, Width).
        # If your noise is flat (Batch, 128), we reshape it to (Batch, 128, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

# cop
class DCDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super(DCDiscriminator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: (batch_size, 3, 32, 32)
            # Output: (batch_size, 64, 16, 16)
            # Note: No BatchNorm on the very first layer of a Discriminator!
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (batch_size, 64, 16, 16)
            # Output: (batch_size, 128, 8, 8)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (batch_size, 128, 8, 8)
            # Output: (batch_size, 256, 4, 4)
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Input: (batch_size, 256, 4, 4)
            # Output: (batch_size, 1, 1, 1)
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            
            # Sigmoid squashes the output into a probability (0.0 to 1.0)
            nn.Sigmoid()
        )

    def forward(self, x):
        # We flatten the final (Batch, 1, 1, 1) output into a simple 1D array of probabilities (Batch, 1)
        return self.net(x).view(-1, 1)

class Module(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # two networks cop vs counterfeiter
        self.generator = DCGenerator(in_channels=in_channels, latent_dim=latent_dim)
        self.discriminator = DCDiscriminator(in_channels=in_channels)

        # Fixed noise for validating/plotting progress
        # register_buffer ensures this moves to the GPU automatically when you do self.to(device)
        self.register_buffer('validation_z', torch.randn(10, latent_dim))

    def forward(self, z):
        return self.generator(z)

    def loss_function(self, y_hat, y):
        # Standard Binary Cross Entropy loss used for GANs
        return F.binary_cross_entropy(y_hat, y)
    
    def plot_imgs(self, epoch = 3, epochs = 10):
        # Generate images from the fixed validation noise
        self.eval() # Set to evaluation mode temporarily
        with torch.no_grad():
            sample_imgs = self(self.validation_z).cpu()
        self.train() # Set back to training mode

        fig = plt.figure()
        for i in range(sample_imgs.size(0)):
            if i == 10:
                break
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            # Assuming channel 0 is grayscale. Adjust if using RGB (3 channels)
            plt.imshow(sample_imgs[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def save_gan_comparison_grid(self, real_images, fixed_noise, epoch):
        """
        Saves a grid of Real Images (top row) vs Fake Images generated from fixed noise (bottom row).
        """
        # 1. Grab up to 10 real images from the current batch
        n = min(real_images.size(0), 10)
        real_subset = real_images[:n]

        # 2. Grab the same number of fixed noise vectors
        noise_subset = fixed_noise[:n]

        # 3. Generate the FAKE images
        self.eval() # Switch to eval mode (crucial if using BatchNorm!)
        with torch.no_grad():
            fake_images = self(noise_subset)
        self.train() # Switch back to training mode immediately

        # 4. Combine the 10 real and 10 fake images into a list of 20 images.
        comparison = torch.cat([real_subset, fake_images])

        # 5. Ensure the results folder exists
        os.makedirs("results", exist_ok=True)

        # 6. Save the grid
        filepath = f"results/dcgan_training_results_{epoch}.png"
        
        # Dataset and models are now in range [-1, 1]. save_image needs [0, 1].
        # normalize=True with value_range=(-1, 1) maps the images correctly.
        save_image(comparison.cpu(), filepath, nrow=n, normalize=True, value_range=(-1, 1))


    def startTraining(self, train_loader, ds_length, learning_rate=2e-4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), epochs=10):
        self.to(device)

        # FIX 1: Use 'self', not 'model'
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        
        # Using the buffer we built earlier for validation!
        fixed_noise = self.validation_z.to(device)

        print("Starting GAN Training Loop...")
        for epoch in range(epochs):
            self.train()
            
            for batch_idx, batch in enumerate(train_loader):
                real_imgs, labels, indices = batch 
                real_imgs = real_imgs.to(device)
                batch_size = real_imgs.size(0)

                # FIX 2: Label Smoothing! Cop is only 90% sure real images are real.
                real_labels = torch.full((batch_size, 1), 0.9, device=device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                # ==========================================
                # 1. TRAIN DISCRIMINATOR
                # ==========================================
                optimizer_D.zero_grad()
                
                # Score Real
                output_real = self.discriminator(real_imgs)
                loss_D_real = self.loss_function(output_real, real_labels)
                
                # Score Fake
                z1 = torch.randn(batch_size, self.latent_dim).to(device)
                fake_imgs1 = self(z1) 
                
                # Detach prevents D from accidentally updating G's weights!
                output_fake = self.discriminator(fake_imgs1.detach())
                loss_D_fake = self.loss_function(output_fake, fake_labels)
                
                loss_D = (loss_D_real + loss_D_fake) / 2
                loss_D.backward()
                optimizer_D.step()

                # ==========================================
                # 2. TRAIN GENERATOR
                # ==========================================
                optimizer_G.zero_grad()
                
                # FIX 3: Fresh Fakes! Give G a brand new attempt against the updated Cop.
                z2 = torch.randn(batch_size, self.latent_dim).to(device)
                fake_imgs2 = self(z2)
                
                # Evaluate against REAL labels (G wants D to think it's 0.9 real!)
                output_fake_for_G = self.discriminator(fake_imgs2)
                loss_G = self.loss_function(output_fake_for_G, real_labels)
                
                loss_G.backward()
                optimizer_G.step()

                # Print progress
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs} [{batch_idx * len(real_imgs)}/{ds_length}] "
                          f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

            # Plot images every 3 epochs
            if (epoch + 1) % 3 == 0:
                # Passing our generator and the validation noise!
                self.save_gan_comparison_grid(real_imgs, fixed_noise, epoch)

        print("Training Complete!")
        return loss_G.item()


    def generate_new_images(self, num_images=10, latent_dim=128, device=None, return_images=False, **kwargs):
        if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # GAN HACK to avoid overly gray pictures: Keep it in train mode so BatchNorm calculates fresh, high-contrast colors!
        self.eval()
        
        with torch.no_grad():
            z = torch.randn(num_images, latent_dim).to(device)
            fake_images = self(z).cpu()

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