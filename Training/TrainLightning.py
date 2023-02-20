# This script needs these libraries to be installed: 
#   torch, torchvision, pytorch_lightning
from BachelorProject.Data_handeling.utils import base_dir
import wandb

import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# Create a class that inherits from LightningModule
class ResnetModels(pl.LightningModule):
    def __init__(self, lr, inp_size, model):
        super().__init__()
        self.lr = lr
        self.inp_size = inp_size
        self.model = model
        # Five output classes
        self.model.fc = nn.Linear(512, 5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        return loss



# init the autoencoder
autoencoder = LitAutoEncoder(lr=1e-3, inp_size=28)

# setup data
batch_size = 32
dataset = MNIST(f'{base_dir}/BachelorProject/Data/MNIST', download=True, transform=ToTensor())
train_loader = utils.data.DataLoader(dataset, shuffle=True)

# initialise the wandb logger and name your wandb project
wandb_logger = WandbLogger(project='Test2WithLightning')

# add your batch size to the wandb config
wandb_logger.experiment.config["batch_size"] = batch_size

# pass wandb_logger to the Trainer 
trainer = pl.Trainer(limit_train_batches=750, max_epochs=5, logger=wandb_logger)

# train the model
trainer.fit(model=autoencoder, train_dataloaders=train_loader)

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()