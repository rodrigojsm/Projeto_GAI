import torch.nn as nn
import torch.nn.functional as F
import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

# Detective: fake or no fake -> 1 output [0, 1]
class DCDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # Input tensor size after conv + pooling on 32x32 images: 20 channels × 5 × 5 = 500
        self.fc1 = nn.Linear(500, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = x.view(x.size(0), -1)  # safer than hardcoding 320

        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return torch.sigmoid(x)

# Generate Fake Data: output like real data [1, 32, 32] and values -1, 1   
class DCGenerator(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100):
        super().__init__()
        self.latent_dim = latent_dim

        self.lin1 = nn.Linear(latent_dim, 8 * 8 * 64)

        self.ct1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 8→16
        self.ct2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 16→32

        # ✅ output channels now = in_channels (1 or 3)
        self.conv = nn.Conv2d(16, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Pass latant space input into linear layer and reshape
        x = self.lin1(x)
        x = F.relu(x)
        x = x.view(-1, 64, 8, 8) # 256

        # Upsample (transposed conv) 16x16 (64 feature maps)
        x = self.ct1(x)
        x = F.relu(x)

        # Upsample to 32x32 (16 feature maps)
        x = self.ct2(x)
        x = F.relu(x)

        # Final conv to get 1 channel output
        return self.conv(x)
    
class Module(pl.LightningModule):
    def __init__(self, in_channels=3, latent_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = DCGenerator(
            in_channels=self.hparams.in_channels,
            latent_dim=self.hparams.latent_dim
        )

        self.discriminator = DCDiscriminator(
            in_channels=self.hparams.in_channels
        )

        self.validation_z = torch.randn(6, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    def training_step(self, batch, batch_idx):
        real_imgs, _, _ = batch
        opt_g, opt_d = self.optimizers()

        # sample noise
        z = torch.randn(real_imgs.shape[0], self.hparams.latent_dim)
        z = z.type_as(real_imgs)

        # generator step
        fake_imgs = self(z)
        y_gen = torch.ones(real_imgs.size(0), 1, device=real_imgs.device)
        g_loss = self.adversarial_loss(self.discriminator(fake_imgs), y_gen)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()

        # discriminator step
        y_real = torch.ones(real_imgs.size(0), 1, device=real_imgs.device)
        y_fake = torch.zeros(real_imgs.size(0), 1, device=real_imgs.device)

        real_loss = self.adversarial_loss(self.discriminator(real_imgs), y_real)
        fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), y_fake)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()

        self.log('g_loss', g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('d_loss', d_loss, prog_bar=True, on_step=True, on_epoch=True)
        return {'loss': g_loss + d_loss}



    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.lin1.weight)
        sample_imgs = self(z).cpu()

        print('epoch ', self.current_epoch)
        fig = plt.figure()
        for i in range (sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs.detach()[i, 0, :, :], cmap='gray_r', interpolation='none')
            plt.title("Generated Data")
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def on_epoch_end(self):
        self.plot_imgs()