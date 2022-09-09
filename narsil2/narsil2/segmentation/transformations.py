import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from skimage.util import random_noise
from skimage.transform import rescale, resize, rotate
from skimage.filters import gaussian
from narsil2.segmentation.utils_omni import labels_to_flows_cpu_omni, normalize99
import cv2
import edt
import time
import random

def normalize99(img, omni=False, invert=False, lower=0.01, upper=99.99):
    img = img.astype(np.float32)
    img_copy = img.copy()
    return np.interp(img_copy, (np.percentile(img_copy, lower), np.percentile(img_copy, upper)), (0, 1))


def all_transforms(phase_imgs, label_imgs, size=(320, 320), scale=(0.5, 1.0), ratio=(3/4., 4/3.),
            rotation_angle=[-90, 90]):

    #phase_imgs = phase_imgs.astype('float32')
    #phase_imgs = normalize99(phase_imgs)
    #label_imgs = label_imgs.astype('float32')
    mean_phase = np.mean(phase_imgs)
    std_phase = np.std(phase_imgs)
    phase_imgs = (phase_imgs - mean_phase) / std_phase


    phase_imgs = TF.to_pil_image(phase_imgs.astype('float32'))
    label_imgs = TF.to_pil_image(label_imgs.astype('float32'))

    angle = transforms.RandomRotation.get_params(rotation_angle)
    phase_imgs = TF.rotate(phase_imgs, angle)
    label_imgs = TF.rotate(label_imgs, angle)

    tensorize = transforms.ToTensor()

    phase_tensors = tensorize(phase_imgs)
    label_tensors = tensorize(label_imgs)

    resize_params = transforms.RandomResizedCrop.get_params(phase_tensors, scale=scale, ratio=ratio)

    cropped_phase = TF.resized_crop(phase_tensors, *resize_params, size=size)
    cropped_labels = TF.resized_crop(label_tensors, *resize_params, size=size, interpolation=transforms.InterpolationMode.NEAREST)

    return cropped_phase.float(), cropped_labels.float()

class UnetTransformations:

    def __init__(self, size=(320, 320), scale=(0.5, 1.0), ratio=(3/4., 4/3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        phase_img = sample['phase']
        label_img = sample['labels'] > 0 # you need to binarize the image somewhere, here we do it, as
        # input images are in labelled format

        t_phase, t_label = all_transforms(phase_img, label_img, self.size, self.scale, self.ratio)

        return {
            'phase': t_phase,
            'labels': t_label,
            'filename': sample['filename']
        }





class OmniTransformations:
    
    def __init__(self, final_size=(320, 320),
                 scale_range=1.0, gamma_range=0.5,
                 return_tensors=False):
        self.final_size = final_size # (H, W)
        self.scale_range = scale_range
        self.gamma_range = gamma_range
        self.return_tensors = return_tensors
        
    def __call__(self, sample):
        #np.random.seed(int(time.time()))
        #print(f"Random state: {np.random.get_state()}")
        H, W = sample['phase'].shape
        phase = sample['phase']
        labels = sample['labels']
    
        phase_t = np.zeros((1, *self.final_size), np.float32)
        labels_t = np.zeros((8, *self.final_size), np.float32)
        # scale_range 
        ds = self.scale_range/2
        scale = np.random.uniform(low=1-ds, high=1+ds, size=2)
        # gamma 
        dg = self.gamma_range/2
        # angle
        theta = np.random.rand() * np.pi * 2
        
        # random translation, take the difference between the scaled dimensions and the crop dimensions
        dxy = np.maximum(0, np.array([W*scale[1]-self.final_size[1], H*scale[0]-self.final_size[0]]))
        # multiplies by a pair of random numbers from -.5 to .5 (different for each dimension) 
        dxy = (np.random.rand(2,) - .5) * dxy 

        # create affine transform
        cc = np.array([W/2, H/2])
        # xy are the sizes of the cropped image, so this is the center coordinates minus half the difference
        cc1 = cc - np.array([W-self.final_size[1], H-self.final_size[0]])/2 + dxy
        # unit vectors from the center
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        # transformed unit vectors
        pts2 = np.float32([cc1,
                cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)
        
        # phase interpolation method
        phase_interp_method  = cv2.INTER_LINEAR
        # label interpolation method
        label_interp_method = cv2.INTER_NEAREST
        
        ##########################################################
        ################## Phase augmentations ###################
        ##########################################################
        phase_intermediate = cv2.warpAffine(phase, M,
                                            (self.final_size[1], self.final_size[0]),
                                            borderMode=0, flags=phase_interp_method)
        # gamma augmentation
        gamma = np.random.uniform(low=1-dg,high=1+dg)
        phase_t = phase_intermediate ** gamma
        
        # percentile clipping
        dp=10
        dpct = np.random.triangular(left=0, mode=0, right=dp, size=2)
        phase_t = normalize99(phase_t, upper=100-dpct[0], lower=dpct[1])
        
        # noise augmentation
        phase_t = random_noise(phase_t, mode="poisson")            
        
        ###########################################################
        ################### Label augmentations ###################
        ###########################################################
        
        for i in range(8):
            labels_t[i] = cv2.warpAffine(labels[i], M,
                                        (self.final_size[1], self.final_size[0]),
                                        borderMode=0, 
                                        flags=label_interp_method
                                       )
        
        # recompute boundaries
        mask = labels_t[6]
        l = labels_t[0].astype(int)
        dist = edt.edt(l, parallel=8)
        labels_t[5] = dist == 1
        
        # rotate the flow vector by an angle, that is used in the transformation
        flow_y_copy = labels_t[2].copy()
        flow_x_copy = labels_t[3].copy()
        dy = (-flow_x_copy * np.sin(-theta) + flow_y_copy * np.cos(-theta))
        dx = (flow_x_copy * np.cos(-theta) + flow_y_copy * np.sin(-theta))
        labels_t[2] = 5.0 * dy * mask
        labels_t[3] = 5.0 * dx * mask
        
        _, _, smooth_distance, _ = labels_to_flows_cpu_omni(l)
        smooth_distance[dist<=0] = -5.0
        labels_t[1] = smooth_distance
        
        bg_edt = edt.edt(mask < 0.5, black_border=True)
        cutoff = 9
        labels_t[7]  = (gaussian(1 - np.clip(bg_edt, 0, cutoff)/cutoff, 1) + 0.5)
        
        
        # convert everything to tensor if return_tensors == True
        if self.return_tensors == True:
            return {
                'phase': torch.from_numpy(phase_t[np.newaxis,].astype('float32')),
                'labels': torch.from_numpy(labels_t),
                'filename': sample['filename']
            }
        else:
            return {
                'phase': phase_t,
                'labels': labels_t,
                'filename': sample['filename']
            }

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

class unpadNumpyArray:

    def __init__(self):
        pass

    def __call__(self, image, shape):
        H, W = shape

        return image[..., :H, :W]

class OmniTestTransformations:

    def __init__(self, return_tensors=False, pad_to=16, shrink_half=False):
        self.return_tensors = return_tensors
        self.pad_to = pad_to
        self.shrink_half = shrink_half

    def __call__(self, sample):

        H, W = sample['phase'].shape

        # pad to nearest multiple of 8 (default, or whatever needed for the U-net)
        pad_operation = padNumpyArray(pad_to=self.pad_to)
        phase_t = pad_operation(sample['phase'])

        # transformed phase
        phase_t = normalize99(phase_t)

        if self.shrink_half:
            phase_t = rescale(phase_t, 0.5, preserve_range=True)

        if self.return_tensors == True:
            return {
                'phase': torch.from_numpy(phase_t[np.newaxis,].astype('float32')),
                'filename': sample['filename'],
                'raw_shape': (H, W)
            }
        else:
            return {
                'phase': phase_t,
                'filename': sample['filename'],
                'raw_shape': (H, W)
            }

class UnetTestTransformations:

    def __init__(self, return_tensors=False, pad_to=16, shrink_half=False,
            add_noise=True, flip=False):
        self.return_tensors = return_tensors
        self.pad_to = pad_to
        self.shrink_half = shrink_half
        self.add_noise = add_noise
        self.flip = flip

    def __call__(self, sample):

        H, W = sample['phase'].shape
        # mean and std subtraction
        mean_phase = np.mean(sample['phase'])
        std_phase = np.std(sample['phase'])
        phase_t = (sample['phase'] - mean_phase) / std_phase

        pad_operation = padNumpyArray(pad_to=self.pad_to)
        phase_t = pad_operation(phase_t)

        if self.add_noise:
            rand_num = np.random.normal(0, 0.15, phase_t.shape)
            phase_t = phase_t + rand_num

        if self.shrink_half:
            phase_t = rescale(phase_t, 0.5, preserve_range=True)
 
        if self.flip:
            phase_t = rotate(phase_t, angle = 180)
       
        
        if self.return_tensors == True:
            return {
                'phase': torch.from_numpy(phase_t[np.newaxis, ].astype('float32')),
                'filename': sample['filename'],
                'raw_shape': (H, W)
            }
        else:
            return {
                'phase': phase_t,
                'filename': sample['filename'],
                'raw_shape': (H, W)
            }



class changedtoPIL(object):

    def __call__(self, sample):
        phaseImg = sample['phase'].astype('int32')
        maskImg = sample['mask']
        weightsImg = sample['weights'].astype('float32')

        sample['phase'] = TF.to_pil_image(phaseImg)
        sample['mask'] = TF.to_pil_image(maskImg)
        sample['weights'] = TF.to_pil_image(weightsImg)
        
        return sample

class randomCrop(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        
        i, j, h, w = transforms.RandomCrop.get_params(sample['phase'], output_size=(self.output_size, self.output_size))

        sample['phase'] = TF.crop(sample['phase'], i, j, h, w)
        sample['mask'] = TF.crop(sample['mask'], i, j, h, w)
        sample['weights'] = TF.crop(sample['weights'], i, j, h, w)
        return sample

    

class randomRotation(object):

    def __init__(self, rotation_angle):
        self.rotation_angle = rotation_angle

    def __call__(self, sample):

        angle = transforms.RandomRotation.get_params(self.rotation_angle)
        sample['phase'] = TF.rotate(sample['phase'], angle)
        sample['mask'] = TF.rotate(sample['mask'], angle)
        sample['weights'] = TF.rotate(sample['weights'], angle)

        return sample

class randomAffine(object):

    def __init__(self, scale, shear):
        self.scale = scale # something like 0.75-1.25
        self.shear = shear # something like [-30, 30, -30, 30]

    def __call__(self, sample):

        angle, translations, scale, shear = transforms.RandomAffine.get_params(degrees=[0, 0], translate=None, 
                scale_ranges=self.scale, shears=self.shear, img_size=sample['phase'].size)

        sample['phase'] = TF.affine(sample['phase'], angle=angle, translate=translations, scale=scale, shear=shear)
        sample['mask'] = TF.affine(sample['mask'], angle=angle, translate=translations, scale=scale, shear=shear)
        sample['weights'] = TF.affine(sample['weights'], angle=angle, translate=translations, scale=scale, shear=shear)
        return sample

class toTensor(object):

    def __call__(self, sample):
        sample['phase'] = transforms.ToTensor()(np.array(sample['phase']).astype(np.float32))
        sample['mask'] = transforms.ToTensor()(np.array(sample['mask']).astype(np.float32))
        sample['weights'] = transforms.ToTensor()(np.array(sample['weights']).astype(np.float32))

        sample['phaseFilename'] = str(sample['phaseFilename'])
        sample['maskFilename'] = str(sample['maskFilename'])
        sample['weightsFilename'] = str(sample['weightsFilename'])
        return sample

class normalize(object):

    def __call__(self, sample):

        sample['phase'] = (sample['phase'] - torch.mean(sample['phase']))/torch.std(sample['phase'])

        # trying to just increase the darkness of cells
        if random.random() < 0.2:
            indices = sample['phase'] < 1.3
            sample['phase'][indices] -= 0.5


        # mask are 255 for True and 0 for false
        sample['mask'] = sample['mask']/255.0

        #sample['weights'] += 1.0
        

        return sample

class addnoise(object):

    def __init__(self, mean=0.0, std=0.15):
        self.mean = mean
        self.std = std

    def __call__(self, sample):

        sample['phase'] = sample['phase'] + torch.randn(sample['phase'].size()) * self.std + self.mean

        return sample



class resizeOneImage(object):

    def __init__(self, imgResizeShape, imgToNetSize):
        assert isinstance(imgResizeShape, tuple)
        assert isinstance(imgToNetSize, tuple)
        self.imgResizeShape = imgResizeShape
        self.imgToNetSize = imgToNetSize

    def __call__(self, image):

        height, width = image.shape
        # check if imageSize is bigger or equal to the 
        image = np.pad(image, pad_width=((0, self.imgResizeShape[0] - height), (0, self.imgResizeShape[1] - width)), 
                          mode='constant', constant_values = 0.0)
        if self.imgResizeShape[0] != self.imgToNetSize[0] or self.imgResizeShape[1] != self.imgToNetSize[1]:
            # Net size is not same a resized image
            image = resize(image, self.imgToNetSize, anti_aliasing=True, preserve_range=True)
        return image
    


class tensorizeOneImage(object):
    def __init__(self, numUnsqueezes=1):
        self.numUnsqueezes = numUnsqueezes
    def __call__(self, phase_image):
        phase_image = phase_image.astype('float32')
        if self.numUnsqueezes == 1:
            return torch.from_numpy(phase_image).unsqueeze(0)
        elif self.numUnsqueezes == 2:
            return torch.from_numpy(phase_image).unsqueeze(0).unsqueeze(0)


class padTo16(object):

    def __call__(self, sample):
        
        width, height = sample['phase'].size
        pad_width = 16 - (width % 16) 
        pad_height = 16 - (height % 16)

        sample['phase'] = TF.pad(sample['phase'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)
        sample['mask'] = TF.pad(sample['mask'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)
        sample['weights'] = TF.pad(sample['weights'], padding=[0, 0, pad_width, pad_height], padding_mode="constant", fill=0)

        return sample


class shrinkSize(object):

    def __call__(self, sample):

        width, height = sample['phase'].size

        if height >= 1024:
            height = height // 2
        if width >= 2048:
            width = width // 2

        sample['phase'] = TF.resize(sample['phase'], (height, width), interpolation=transforms.InterpolationMode.BILINEAR)
        sample['mask'] = TF.resize(sample['mask'], (height, width), interpolation=transforms.InterpolationMode.NEAREST)
        sample['weights'] = TF.resize(sample['weights'], (height, width), interpolation=transforms.InterpolationMode.BILINEAR)

        return sample


class randomVerticalFlip(object):

    def __call__(self, sample):

        if random.random() < 0.5:
            sample['phase'] = TF.vflip(sample['phase'])
            sample['mask'] = TF.vflip(sample['mask'])
            sample['weights'] = TF.vflip(sample['weights'])

        return sample
