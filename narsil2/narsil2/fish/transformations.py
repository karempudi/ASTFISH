# transformations to do on fish images to match them 
# correctly with the transformations done on phase images
import numpy as np
import torch
from skimage.transform import rescale, resize, rotate
from skimage.filters import gaussian

class padNumpyArray:

    def __init__(self, pad_to=8):
        # pad to nearest multiple of pad_to (typically 8 or 16)
        self.pad_to = pad_to

    def __call__(self, image):
        H, W = image.shape
        
        if H % self.pad_to != 0:
            H_t = ((H // self.pad_to) + 1) * self.pad_to 
        else:
            H_t = H
        if W % self.pad_to != 0:
            W_t = ((W // self.pad_to) + 1) * self.pad_to
        else:
            W_t = W

        image = np.pad(image, pad_width= ((0, H_t - H), (0, W_t - W)),
                             mode='constant')

        return image

class FishTransformations:

    def __init__(self, return_tensors=False, pad_to=16, shrink_half=False,
            flip=False):
    
        self.return_tensors = return_tensors
        self.pad_to = pad_to
        self.shrink_half = shrink_half
        self.flip = flip

    def __call__(self, fish_img):

        H, W = fish_img.shape

        pad_operation = padNumpyArray(pad_to=self.pad_to)
        fish_img_padded = pad_operation(fish_img)

        if self.shrink_half:
            fish_img_padded = rescale(fish_img_padded, 0.5, preserve_range=True)
        
        if self.return_tensors == True:
            return torch.from_numpy(fish_img_padded[np.newaxis, ].astype('float32'))
        else:
            return fish_img_padded 
