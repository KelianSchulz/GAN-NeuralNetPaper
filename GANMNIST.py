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

#-------------------------
# Hyperparameters
#-------------------------
BATCH_SIZE = 32
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
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(-1, 256 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.fc2(x))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut-Verbindung
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        identity = self.shortcut(x)  # Shortcut-Verbindung
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual-Verbindung
        out = self.relu(out)
        return out

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256 * 7 * 7)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Residual Blocks
        self.resblock1 = ResidualBlock(128, 128)
        self.resblock2 = ResidualBlock(64, 64)
        self.resblock3 = ResidualBlock(64, 64)
        
        # Dritter Convolutional Layer
        self.conv3 = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.tanh = nn.Tanh()

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = self.resblock3(x)

        x = self.conv3(x)
        
      

        return self.tanh(x)


#-------------------------
# GAN 
#-------------------------
class GAN(pl.LightningModule):
    def __init__(self, z_dim=100, lr=0.0002):
        super().__init__()
        self.save_hyperparameters()
        self.generator = Generator(z_dim=self.hparams.z_dim).to(device)
        self.discriminator = Discriminator().to(device)
        self.validation_z = torch.randn(6, self.hparams.z_dim).to(device)
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_pred, y):
        return F.binary_cross_entropy(y_pred, y)

    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        real_imgs = real_imgs.to(device)

        z = torch.randn(real_imgs.shape[0], self.hparams.z_dim).to(device)

        # Generator Update
        fake_imgs = self(z)
        fake_imgs = fake_imgs.to(device)
        y_pred = self.discriminator(fake_imgs)

        y = torch.ones(real_imgs.size(0), 1).to(device)

        g_loss = self.adversarial_loss(y_pred, y)

        log_dict = {"g_loss": g_loss}
        # Manuelles Update des Generators
        opt_g = self.optimizers()[0]
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        # Diskriminator Update
        fake_imgs = self(z)
        fake_imgs = fake_imgs.to(device)
        real_pred = self.discriminator(real_imgs)
        fake_pred = self.discriminator(fake_imgs.detach())

        y_real = torch.ones(real_imgs.size(0), 1).to(device)
        y_fake = torch.zeros(real_imgs.size(0), 1).to(device)

        real_loss = self.adversarial_loss(real_pred, y_real)
        fake_loss = self.adversarial_loss(fake_pred, y_fake)

        d_loss = (real_loss + fake_loss) / 2  # d_loss initialisieren

        log_dict["d_loss"] = d_loss
        # Manuelles Update des Diskriminators
        opt_d = self.optimizers()[1]
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        return {"loss": g_loss + d_loss, "progress_bar": log_dict, "log": log_dict}


    def configure_optimizers(self):
        lr = self.hparams.lr
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        return [optimizer_g, optimizer_d], []
    
    def plot_imgs(self):
        z = self.validation_z.type_as(self.generator.fc1.weight).to(device)
        sample_imgs = self(z).cpu()

        # Display generated images and discriminator predictions
        fig = plt.figure(figsize=(10, 10)) 
        for i in range(sample_imgs.size(0)):
            plt.subplot(2, 3, i+1)
            plt.tight_layout()
            plt.imshow(sample_imgs[i, 0, :, :].detach().numpy(), cmap="gray_r", interpolation="none")  # .detach() hier verwenden
            fake_pred = self.discriminator(sample_imgs[i].unsqueeze(0).to(device)).cpu().detach()
            title = f"Fake: {fake_pred.item():.2f}"
            plt.title(title)
            plt.xticks([])
            plt.yticks([])
            plt.axis("off")

        plt.show()


    def on_epoch_end(self):
        # Plot after every epoch
        self.plot_imgs()

    def on_train_end(self):
        # Plot at the end of the training
        self.plot_imgs()
        torch.save(self.generator.state_dict(), 'generator.pth')

# Model and DataModule Initialization
datamodule = MNISTDataModule(data_dir="./data", batch_size=BATCH_SIZE)
model = GAN()

trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, datamodule=datamodule)


