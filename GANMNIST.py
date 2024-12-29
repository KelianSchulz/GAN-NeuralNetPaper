import os
import torch
import torch.optim.adam
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping 
import torch.nn.init as init
import numpy as np
#-------------------------
# Hyperparameters
#-------------------------
BATCH_SIZE = 64
AVAIL_GPUS = min(1, torch.cuda.device_count())


#-------------------------
# Use GPU 
#-------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#-------------------------
# Data Module
#-------------------------
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=BATCH_SIZE):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == "test" or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)


#-------------------------
# Model
#-------------------------

def init_weights_he(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
        if module.bias is not None:
            module.bias.data.fill_(0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)  # InstanceNorm statt BatchNorm
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(64, affine=True)  # InstanceNorm statt BatchNorm
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.in3 = nn.InstanceNorm2d(128, affine=True)  # InstanceNorm statt BatchNorm

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.in4 = nn.InstanceNorm2d(256, affine=True)  # InstanceNorm statt BatchNorm

        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.7)
        self.apply(init_weights_he)

    def forward(self, x):
        x = self.pool(F.relu(self.in1(self.conv1(x))))
        x = self.pool2(F.relu(self.in2(self.conv2(x))))
        x = F.relu(self.in3(self.conv3(x)))
        x = F.relu(self.in4(self.conv4(x)))

        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x))) 
        x = self.fc2(x)  # Sigmoid function weglassen
        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.in1 = nn.InstanceNorm2d(out_channels)  # InstanceNorm statt BatchNorm
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.in2 = nn.InstanceNorm2d(out_channels)  # InstanceNorm statt BatchNorm

        # Shortcut-Verbindung
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = self.shortcut(x)  # Shortcut-Verbindung
        out = self.relu(self.in1(self.conv1(x)))  # InstanceNorm angewendet
        out = self.in2(self.conv2(out))  # InstanceNorm angewendet
        out += identity  # Residual-Verbindung
        out = self.relu(out)
        return out


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.in1 = nn.InstanceNorm2d(128)  # InstanceNorm statt BatchNorm
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.in2 = nn.InstanceNorm2d(64)   # InstanceNorm statt BatchNorm

        # Residual Blocks
        self.resblock1 = ResidualBlock(128, 128)
        self.resblock2 = ResidualBlock(64, 64)
        self.resblock3 = ResidualBlock(64, 64)
        
        # Dritter Convolutional Layer
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()
        self.apply(init_weights_he)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.in1(self.conv1(x)))  # InstanceNorm angewendet
        x = self.resblock1(x)
        x = F.relu(self.in2(self.conv2(x)))  # InstanceNorm angewendet
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.conv3(x)
        return self.tanh(x)


def wasserstein_loss(y_pred, y):
    return torch.mean(y_pred * y)
    


#-------------------------
# GAN 
#-------------------------
class GAN(pl.LightningModule):
    def __init__(self, z_dim=100, lr=0.00005, d_steps=1):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(z_dim=self.hparams.z_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.validation_z = torch.randn(6, self.hparams.z_dim).to(self.device)
        self.automatic_optimization = False  # Manuelle Optimierung

    def validation_step(self, batch, batch_idx):
        
        pass



    def compute_gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        d_interpolates = self.discriminator(interpolates)
        fake = torch.ones(batch_size, 1).to(self.device)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, z):
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(self.device)

        # ===== Diskriminator-Update =====
        z = torch.randn(real_imgs.size(0), self.hparams.z_dim, device=self.device)
        fake_imgs = self(z).detach()  # Generator-Ausgabe (ohne Gradienten)
        fake_imgs = fake_imgs.to(self.device)

        # Diskriminator Vorhersagen
        real_pred = self.discriminator(real_imgs)
        fake_pred = self.discriminator(fake_imgs)

        # Gradient Penalty
        gradient_penalty = self.compute_gradient_penalty(real_imgs, fake_imgs)

        # Gesamtverlust für Diskriminator
        lambda_gp = 5  # WGAN-GP Penalty
        d_loss = -torch.mean(real_pred) + torch.mean(fake_pred) + lambda_gp * gradient_penalty

        # 1 Schritt für den Diskriminator
        opt_d = self.optimizers()[1]
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # ===== Generator-Update =====
        z = torch.randn(real_imgs.size(0), self.hparams.z_dim, device=self.device)
        fake_imgs = self(z)  # Generator-Ausgabe
        fake_pred = self.discriminator(fake_imgs)

        # Generator optimieren (minimiere -D(G(z)))
        g_loss = -torch.mean(fake_pred)

        # 1 Schritt für den Generator
        opt_g = self.optimizers()[0]
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        # ===== Logging =====
        self.log("g_loss", g_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("d_loss", d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("gradient_penalty", gradient_penalty, prog_bar=True, on_step=True, on_epoch=True)

        return {"loss": g_loss + d_loss}



    def configure_optimizers(self):
        # Separate Learning Rates für Generator und Diskriminator
        lr_g = 0.0001  # Lernrate für den Generator
        lr_d = 0.00005  # Lernrate für den Discriminator

        # Optimizer definieren
        optimizer_g = torch.optim.AdamW(self.generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.AdamW(self.discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

         # Rückgabe der Optimizer
        return [optimizer_g, optimizer_d], []  # Wir geben eine Liste von Optimierern zurück

    def plot_imgs(self, epoch, num_images=20):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        z = torch.randn(num_images, self.hparams.z_dim).to(device)
        sample_imgs = self(z).cpu()

        rows = int(np.sqrt(num_images))
        cols = int(np.ceil(num_images / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(num_images):
            axes[i].imshow(sample_imgs[i, 0, :, :].detach().numpy(), cmap="gray_r", interpolation="none")
            fake_pred = self.discriminator(sample_imgs[i].unsqueeze(0).to(device)).cpu().detach()
            axes[i].set_title(f"Fake: {fake_pred.item():.2f}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()

        

class GANCallback(pl.Callback):
    def on_train_end(self, trainer, pl_module):
        # Bilder am Ende des Trainings generieren
        print("Training beendet - Generiere finale Bilder")
        pl_module.plot_imgs('final')
        torch.save(pl_module.generator.state_dict(), 'Wassersteingenerator_final.pth')
        torch.save(pl_module.discriminator.state_dict(), 'WassersteinDiscriminator_final.pth')

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        print(f"Current Epoch: {epoch}")  # Debug-Ausgabe, um die aktuelle Epoche zu überprüfen
        if epoch % 25 == 0:  # Sicherstellen, dass auch bei Epoche 0 gespei  chert wird
            print(f"Saving model and images at epoch {epoch}")  # Debug-Ausgabe
            pl_module.plot_imgs(epoch)
            torch.save(pl_module.generator.state_dict(), f'Wassersteingenerator_epoch_{epoch}.pth')
            torch.save(pl_module.discriminator.state_dict(), f'WassersteinDiscriminator_epoch_{epoch}.pth')



# Model und DataModule Initialisierung
datamodule = MNISTDataModule(data_dir="./data", batch_size=BATCH_SIZE)
model = GAN()

early_stopping = EarlyStopping(
    monitor="g_loss",  
    patience=20,       # Stoppt, wenn sich der Verlust 10 Epochen lang nicht verbessert
    mode="min",    
)

trainer = pl.Trainer(
    max_epochs=200,    # Maximale Anzahl der Epochen
    callbacks=[early_stopping, GANCallback()],  
)

trainer.fit(model, datamodule=datamodule)