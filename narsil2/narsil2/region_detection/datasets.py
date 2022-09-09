# dataset loader for yolo net
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image, ImageFile
from utils import iou_width_height 

class YOLODataset(Dataset):
    
    def __init__(self, 
                 csv_file,
                 img_dir,
                 label_dir,
                 anchors,
                image_size=416,
                S=[13, 26, 52],
                C=20,
                transform=None):
        
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.Tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # apply augmentations with albumentations
        if self.transform:
            augmentations = self.transform(image=image, bboxes = bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
            
        # building the target below
        # 6 targets are object_score, X, Y, H, W, class_number
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3 # each scale should have one anchor
            for anchor_idx in anchor_indices:
                # get the hight
                #scale_idx = anchor_idx  // self.num_anchors_per_scale
                scale_idx  = torch.div(anchor_idx, self.num_anchors_per_scale, rounding_mode='floor')
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                # get the center of the coordinates at that scale level
                i, j = int(S * y), int(S * x) # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                #print(i, j)
                if not anchor_taken and not has_anchor[scale_idx]:
                    # marking there is an object at that scale index
                    # by setting object score to 1
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    
                    x_cell, y_cell = S * x -j, S * y - i
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )
                    
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    #print(anchor_on_scale)
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # ignore prediction
        
        return image, tuple(targets)

