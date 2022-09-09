# Pytorch lightning modules for running train and eval loop with
# checkpointing and saving
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl

from networks import model_dict
from losses import loss_dict


def create_model(model_name, model_hparams):
    if model_name in model_dict:
        return model_dict[model_name](**model_hparams)
    else:
        assert False, f"Unknown model name \"{model_name}\" Available models are: {str(model_dict.keys())}"

def create_loss(loss_name, loss_hparams):
    if loss_name in loss_dict:
        return loss_dict[loss_name](**loss_hparams)
    else:
        assert False, f"Unknown loss function \"{loss_name}\". Available: {str(model_dict.keys())}"

class YOLOTrainingModule(pl.LightningModule):
    
    def __init__(self, model_name, model_hparams,
            loss_name, loss_hparams, 
            optimizer_name, optimizer_hparams, scheduler_hparams, device, 
            scaled_anchors):
        super().__init__()
        # All hyper parameters are bundle together with the model for reloading
        self.save_hyperparameters()
        # Create the model
        self.model = create_model(model_name, model_hparams)
        # Loss function
        self.loss_module = create_loss(loss_name, loss_hparams)
        # gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        self.scaled_anchors = scaled_anchors.to(device)

        self.automatic_optimization = False

    def forward(self, images):
        # Forward function that will run when the model is called
        # model will take images and return 3 tensor, one for each scale
        return self.model(images)


    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name =="SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer:"
            
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 15], gamma=0.1)
        
        return [optimizer],  [scheduler]
    
    
    def training_step(self, batch, batch_idx):
        # batch is the out put fo the training data loader
        # images is a batch of images and targets are 3 tensors 
        # corresponding to each scale
        optimizer = self.optimizers()
        optimizer.zero_grad()
        images, targets = batch

        with torch.cuda.amp.autocast():
            preds = self.model(images)
            loss = (
                    self.loss_module(preds[0], targets[0], self.scaled_anchors[0])
                    + self.loss_module(preds[1], targets[1], self.scaled_anchors[1])
                    + self.loss_module(preds[2], targets[2], self.scaled_anchors[2])
                )

        train_acc = 0.0
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()
        
        # Logs stuff to tensor board
        #self.log('train_acc', train_acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # batch is the out put fo the training data loader
        # images is a batch of images and targets are 3 tensors 
        # corresponding to each scale
        #optimizer = self.optimizers()
        #optimizer.zero_grad()
        images, targets = batch

        with torch.cuda.amp.autocast():
            preds = self.model(images)
            validation_loss = (
                    self.loss_module(preds[0], targets[0], self.scaled_anchors[0])
                    + self.loss_module(preds[1], targets[1], self.scaled_anchors[1])
                    + self.loss_module(preds[2], targets[2], self.scaled_anchors[2])
                    )

        val_acc = 0.0

        #self.scaler.scale(validation_loss).backward()
        #self.scaler.step(optimizer)
        #self.scaler.update()
 
        # logs stuff to tensorboard
        self.log('val_loss', validation_loss, on_epoch=True)
        #self.log('val_acc', val_acc)
    
    def test_step(self, batch, batch_idx):
        # batch is the out put fo the training data loader
        # images is a batch of images and targets are 3 tensors 
        # corresponding to each scale
        #optimizer = self.optimizers()
        #optimizer.zero_grad()
        images, targets = batch

        with torch.cuda.amp.autocast():
            preds = self.model(images)
            test_loss = (
                    self.loss_module(preds[0], targets[0], self.scaled_anchors[0])
                    + self.loss_module(preds[1], targets[1], self.scaled_anchors[1])
                    + self.loss_module(preds[2], targets[2], self.scaled_anchors[2])
                    )

        test_acc = 0.0

        #self.scaler.scale(test_loss).backward()
        #self.scaler.step(optimizer)
        #self.scaler.update()
 
        # logs stuff to tensorboard
        self.log('test_loss', test_loss, on_epoch=True)
        #self.log('val_acc', val_acc)
    
