import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from narsil2.segmentation.losses import loss_dict
from narsil2.segmentation.networks import model_dict
import numpy as np

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


# from the UVA course, transformer tutorial.
# at https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
#
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class UTrainingModule(pl.LightningModule):
    
    def __init__(self, model_name, model_hparams, 
                 loss_name, loss_hparams,
                 optimizer_name, optimizer_hparams, scheduler_hparams):
        super().__init__()
        # All hyper parameters are bundled together with the model, for reloading
        self.save_hyperparameters()
        # Create the model
        self.model = create_model(model_name, model_hparams)
        # Loss function
        self.loss_module = create_loss(loss_name, loss_hparams)
        #self.example_input_array = torch.zeros((1, 1, 320, 320), dtype=torch.float32)
        
    def forward(self, imgs):
        # Forward function that will run when the model is called
        return self.model(imgs)

    
    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.Adam(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "AdamW":
            optimizer = optim.Adam(
                self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name =="SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer:"
            
        #scheduler = optim.lr_scheduler.MultiStepLR(
        #    optimizer, milestones=[100, 15], gamma=0.1)
        
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             **self.hparams.scheduler_hparams)
        return [optimizer],  [{'scheduler': lr_scheduler, 'interval': 'step'}]
        #return optimizer
    
    def training_step(self, batch, batch_idx):
        # batch is the out put fo the training data loader
        phase, labels = batch['phase'], batch['labels']
        batch_size = phase.shape[1]
        preds = self.model(phase)
        loss = self.loss_module(preds, labels)
        train_acc = 0.0
        
        # Logs stuff to tensor board
        #self.log('train_acc', train_acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        phase, labels = batch['phase'], batch['labels']
        batch_size = phase.shape[1]
        preds = self.model(phase)
        validation_loss = self.loss_module(preds, labels)
        val_acc = 0.0
        
        # logs stuff to tensorboard
        self.log('val_loss', validation_loss, on_epoch=True, batch_size=batch_size)
        #self.log('val_acc', val_acc)
    
    def test_step(self, batch, batch_idx):
        phase, labels = batch['phase'], batch['labels']
        batch_size = phase.shape[1]
        preds = self.model(phase)
        test_loss = self.loss_module(preds, labels)
        test_acc = 0.0
        
        # logs stuff to tensorboard
        self.log('test_loss', test_loss, on_epoch=True, batch_size=batch_size)
        #self.log('test_acc', test_acc)