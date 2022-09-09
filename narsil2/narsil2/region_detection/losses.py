# Loss functions for YOLOv3

import random
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        # constants for weighting differenet losses differently
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        
    def forward(self, predictions, target, anchors):
        
        obj = target[..., 0] == 1
        no_obj = target[..., 0] == 0
        
        # object score loss for no objects
        no_object_loss = self.bce(
            (predictions[..., 0:1][no_obj]), (target[..., 0:1][no_obj])
        )
        
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), 
                               torch.exp(predictions[..., 3:5]) * anchors], dim=-1)

        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        # object score loss when there are object, and object probability is the iou
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        
        # x, y, h, w mse loss
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )
        # compute mse loss for boxes
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
        # class loss
        # you have a list of proabilities on one side, and an integer of the class 
        # number on the other side in cross entropy calculation
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
        
loss_dict = {
    "YoloLoss": YoloLoss
}