# Lightning module for the tracker training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from losses import loss_dict
from networks import model_dict
import numpy as np


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\" Available models are : {str(model_dict.keys())}"

def create_loss(loss_name, loss_hparams):
    if loss_name in loss_dict:
        return loss_dict[loss_name](**loss_hparams)
    else:
        assert False, f"Unknown loss function \"{loss_name}\". Available: {str(model_dict.keys())}"

class trackerTrainingModule(pl.LightningModule):

    def __init__(self, model_name, model_hparams,
                 loss_name, loss_hparams,
                 optimizer_name, optimizer_hparams,
                 scheduler_hparams):
        super().__init__()
        # All hyperparameters are bundled together with the model, for reloading
        self.save_hyperparameters()
        # Create a model
        self.model = create_model(model_name, model_hparams)
        # Loss function
        self.loss_module = create_loss(loss_name, loss_hparams)
    
    def forward(self, imgs):
        # Forward function that will run when the model is called
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(
                self.parameters(), **self.hparams.optimizer_hparams
            )
        
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, **self.hparams.scheduler_hparams
        )
        
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        # batch is the output of the training data loader

        self.log('train_loss', 0.0, on_epoch=True)
        return 0.0

    def validation_step(self, batch, batch_idx):

        self.log('val_loss', 0.0, on_epoch=True)
        return 0.0
    
    def test_step(self, batch, batch_idx):

        self.log('test_loss', 0.0, on_epoch=True)

