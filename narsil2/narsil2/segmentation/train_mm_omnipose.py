
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import ncolor, edt
from pathlib import Path
np.seterr(invalid='ignore')

from narsil2.segmentation.datasets import MMData, MMDataTest
from narsil2.segmentation.utils_omni import reconstruct_masks_cpu_omni, format_labels, clean_boundary
from narsil2.segmentation.transformations import OmniTestTransformations, unpadNumpyArray, OmniTransformations
from narsil2.segmentation.networks import Unet
from narsil2.segmentation.losses import OmniLoss
from narsil2.segmentation.lightning_modules import UTrainingModule
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from skimage import io


def retrain_model(model_name, max_epochs=20, train_from_scratch=True, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model to run
        save_name (optional) - This name will be used in checkpoint and logging directory
    """
    if save_name is None:
        save_name = model_name
        
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
                        gpus=[1],
                        max_epochs=max_epochs,
                        callbacks=[ModelCheckpoint(save_weights_only=True, mode="min",
                                                   monitor="train_loss"),
                                   LearningRateMonitor("epoch"),
                                   TQDMProgressBar(refresh_rate=10)],
                        log_every_n_steps=10
                        )
    trainer.logger._log_graph = False
    trainer.logger._default_hp_metric = None
    
    # pretrained, then load and skip it
    pretrained_filename = MODEL_CHECKPOINT_PATH
    if train_from_scratch == False and os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading ...")
        model = UTrainingModule(model_name=model_name, **kwargs)
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        trainer.fit(model, train_loader, test_loader)
        model = UTrainingModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
    else:
        model = UTrainingModule(model_name=model_name, **kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = UTrainingModule.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
        
    # test for accuracy/ whatever is logged. we only log losses for now.
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    
    result = {"train": train_result[0]["test_loss"], "test": test_result[0]["test_loss"]}
    
    return model, result


DATASET_PATH = "../../../data/omnipose_data"
# Path to the directory where the pretrained models are saved
CHECKPOINT_PATH = "../../../saved_models/"

MODEL_CHECKPOINT_PATH = CHECKPOINT_PATH + 'None.ckpt'

#SAVED_MODEL_PATH = "../saved_models/pth_files/unet_omni.pth"

GPU_NUMBER = 1


device = torch.device("cuda:" + str(GPU_NUMBER)) if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

net_size=320
batch_size = 16


train_data = MMData(train=True, dataset_path=DATASET_PATH,
                   flows=True, saved_flows=True, phase_fileformat='*.tif',
                   labels_fileformat="*.npy", mothermachine_data=True,
                   transforms=OmniTransformations(final_size=(net_size, net_size), return_tensors=True))

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False,
                          drop_last=False, pin_memory=True, num_workers=4)


test_data = MMData(train=True, dataset_path=DATASET_PATH,
                   flows=True, saved_flows=True, phase_fileformat='*.tif',
                   labels_fileformat="*.npy", mothermachine_data=True,
                   transforms=OmniTransformations(final_size=(net_size, net_size), return_tensors=True))

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)


max_epochs = 1000
model, result = retrain_model(
            "Unet", 
            max_epochs=max_epochs,
            save_name="Omnipose",
            model_hparams={
                "channels_by_scale": [1, 32, 64, 128, 256],
                "num_classes": 4,
                "upsample_type":"transposeConv",
                "feature_fusion_type":"concat",
            }, 
            loss_name="Omni",
            loss_hparams={"gpu": 1},
            optimizer_name="SGD",
            optimizer_hparams={"lr": 0.05, "momentum": 0.9, "weight_decay": 1e-4},
            #optimizer_hparams={"lr": 0.05},
            scheduler_hparams={"warmup": 20, "max_iters": max_epochs*30}
)