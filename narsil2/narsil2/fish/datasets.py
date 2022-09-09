# File for bundling the FISH data and bounding box detections on the FISH images
# 
from scipy.ndimage.morphology import distance_transform_edt
from pathlib import Path
import numpy as np
import glob
import os
from skimage.io import imread
from skimage.transform import rotate
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from joblib import dump, load
from skimage.exposure import equalize_adapthist

MULTI_LABELS_TO_SPECIES = {
    0: 'ecoli',
    1: 'kpneumoniae',
    2: 'paeruginosa',
    3: 'efaecalis',
    4: 'abaumanii',
    5: 'pmirbalis',
    6: 'saureus',
    7: 'background'
}

MULTI_SPECIES_TO_LABELS =  {
    'ecoli': 0,
    'kpneumoniae': 1,
    'paeruginosa': 2,
    'efaecalis': 3,
    'abaumanii': 4,
    'pmirbalis': 5,
    'saureus': 6,
    'background': 7
}

SPECIES_TO_FLUOR_MAP = {
    'ecoli': ['cy3'],
    'kpneumoniae': ['cy3', 'cy5'],
    'paeruginosa': ['alexa488', 'texasred'],
    'efaecalis': ['alexa488', 'cy5'],
    'abaumanii': ['alexa488', 'cy3'],
    'pmirbalis': ['alexa488'],
    'saureus': ['cy5', 'texasred'],
}

class singleColourFISHData(object):

    def __init__(self, fish_dirname, channel_names, channel_size=(120, 1800),
            fluor_channel_threshold=None, min_box_height=90,
            transforms=None, fileformat = '.tiff', background_fishdata = None,
            flip=False, force=False):

        if type(fish_dirname) == str:
            self.fish_dirname = Path(fish_dirname)
        else:
            self.fish_dirname = fish_dirname

        self.channel_names = channel_names
        self.fileformat = fileformat

        self.fish_images = {}
        self.fish_boxes = {}
        self.fluor_channel_threshold = fluor_channel_threshold # a dict with keys as channel names

        self.channel_size = channel_size # top and bottom to remove stuff outside the channel
        self.transforms = transforms
        self.min_box_height = min_box_height

        self.background_fishdata = background_fishdata

        self.flip = flip

        # read files and construct bboxes
        for channel in self.channel_names:
            filename = self.fish_dirname / (channel + '.tiff')
            image = imread(filename, as_gray=True).astype('float32')

            if self.flip:
                #image = rotate(image, angle=180, preserve_range=True) 
                image = np.flipud(image)

            if transforms == 'box':
                image = image - self.background_fishdata[channel]
                image = gaussian_filter(image, sigma=3)

                if np.sum(image) == 0:
                    self.fish_images[channel] = self.background_fishdata[channel]
                    self.fish_boxes[channel] = []
                else:
                    image[:self.channel_size[0], :] = 0
                    image[self.channel_size[1]:, :] = 0
                    if force == False:
                        image_bool = image > max(threshold_otsu(image), self.fluor_channel_threshold[channel])
                    else:
                        image_bool = image > self.fluor_channel_threshold[channel]

                    boxes = []
                    y1 = 10
                    y2 = image.shape[1] - 10
                    width = y2 - y1 + 1
                    xlims_bool = np.sum(image_bool, axis=1) > (width/3.5)
                    xlims = np.where(np.diff(xlims_bool) == 1)[0]

                    if len(xlims)%2 == 1 and len(xlims) != 0:
                        xlims = xlims[:-1]
                    
                    for i in range(0, len(xlims), 2):
                        xy = (y1, xlims[i])
                        height = xlims[i+1] - xlims[i] + 1
                        if height >= self.min_box_height:
                            boxes.append((xy, width, height))
                    
                    if len(boxes) != 0:
                        self.fish_boxes[channel] = boxes
                    else:
                        self.fish_boxes[channel] = []
                    
                    self.fish_images[channel] = image
            else:
                self.fish_images[channel] = image


    def __len__(self):
        return len(self.channel_names)

    def __getitem__(self, channel):
        if channel not in self.channel_names:
            return None
        else:
            return self.fish_images[channel]

    def plot(self, channel):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(self.fish_images[channel], cmap='gray')
        if channel in self.fish_boxes.keys():
            for box in self.fish_boxes[channel]:
                ax.add_patch(Rectangle(*box, linewidth=1, edgecolor='r', facecolor='none'))
        ax.set_title(f"{channel} Image")
        plt.show()

    def get_number_fluor_channels(self):
        num_channels = 0
        for channel in self.channel_names:
            if len(self.fish_boxes[channel]) > 0:
                num_channels += 1
        return num_channels

    def get_fluor_channels(self):
        fluor_channels = []
        for channel in self.channel_names:
            if len(self.fish_boxes[channel]) > 0:
                fluor_channels.append(channel)
        return fluor_channels

    def plot_all_channels(self):
        fig, ax = plt.subplots(nrows = 1, ncols = len(self.channel_names))
        for i, channel in enumerate(self.channel_names, 0):
            ax[i].imshow(self.fish_images[channel], cmap='gray')
            if self.transforms == 'box' and len(self.fish_boxes[channel]) != 0:
                for box in self.fish_boxes[channel]:
                    ax[i].add_patch(Rectangle(*box, linewidth=1, edgecolor='r', facecolor = 'none'))
            ax[i].set(title=channel)
        plt.show(block=False)
        

class multiColorFISHData(object):

    def __init__(self, fish_dirname, channel_names, classifier_filename, channel_size=(120, 1800), 
            fileformat='.tiff', bg_normalization_vector=None, flip=False, min_box_height=90, smooth=True,
            species_to_fluor_map=None, signal_width_thres=0.35):
        
        if type(fish_dirname) == str:
            self.fish_dirname = Path(fish_dirname)
        else:
            self.fish_dirname = fish_dirname
        
        self.channel_names = channel_names
        self.fileformat = fileformat

        self.fish_images = {}
        self.fish_boxes = {}
        
        self.channel_size = channel_size
        self.flip = flip
        self.smooth=smooth
        self.min_box_height = min_box_height
        self.transforms='box' # just for compatibility with single color plots in the track bundle

        self.classifier_filename = classifier_filename
        self.clf = load(self.classifier_filename)
        self.bg_normalization_vector = bg_normalization_vector
        self.signal_width_thres = signal_width_thres

        for channel in self.channel_names:
            filename = self.fish_dirname / (channel + '.tiff')
            image = imread(filename)
            self.dtype = image.dtype
            self.height, self.width = image.shape

            if self.flip:
                image = np.flipud(image)
            
            if self.smooth:
                image = gaussian_filter(image, sigma=5)
            
            self.fish_images[channel] = image
            self.fish_boxes[channel] = []

        # now loop and construct fishboxes and put them in self.fish_boxes
        self.images_stack = [self.fish_images[channel].copy() for channel in self.channel_names]
        self.images_stack = np.stack(self.images_stack, axis=0)
        #print(self.images_stack.shape)
        #reshape for the classfier
        images_stack_reshaped = self.images_stack.reshape(len(self.channel_names), -1)
        # normalization by the normalization vector # subtracting the autofluorescence of PDMS
        images_stack_reshaped = images_stack_reshaped - self.bg_normalization_vector[:, None] 
        images_stack_reshaped = images_stack_reshaped.T
        #print(images_stack_reshaped.shape)
        # classify the species 
        if self.clf is not None:
            classes_predicted = self.clf.predict(images_stack_reshaped)
            classes_predicted_reshaped = classes_predicted.reshape(self.height, self.width)
            #print(classes_predicted_reshaped.shape)
        
        self.predicted_species_map = classes_predicted_reshaped

        height, width = self.predicted_species_map.shape

        line = np.ones((height, )) * MULTI_SPECIES_TO_LABELS['background']

        for i in range(height):
            uniq, counts = np.unique(self.predicted_species_map[i, :], return_counts=True)
            largest_count_class_idx = np.argmax(counts * (uniq != MULTI_SPECIES_TO_LABELS['background']))
            class_label = uniq[largest_count_class_idx]
            class_counts = counts[largest_count_class_idx]
            if class_counts >= self.signal_width_thres * width:
                line[i] = class_label
            else:
                line[i] = MULTI_SPECIES_TO_LABELS['background']

        points_of_change = np.where(np.diff(line) != 0)[0]
        self.species_bboxes = {}
        if len(points_of_change) >= 2:
            for a, b in zip(points_of_change[:-1], points_of_change[1:]):
                if b - a > 1 and b - a > self.min_box_height:
                    species_number = int(np.mean(line[a+1:b]))
                    if species_number != MULTI_SPECIES_TO_LABELS['background']:
                        if MULTI_LABELS_TO_SPECIES[species_number] not in self.species_bboxes:
                            self.species_bboxes[MULTI_LABELS_TO_SPECIES[species_number]] = [((20, a), 40, b - a)]
                        else:
                            self.species_bboxes[MULTI_LABELS_TO_SPECIES[species_number]].append(((20, a), 40, b - a))

        # walk down and draw bboxes on the maximum species and cap the min bbox size
        
        # look at the species and do bboxes on the channels, # this is for backward compatibility
        for species, bboxes_list in self.species_bboxes.items():
            channels_of_species = SPECIES_TO_FLUOR_MAP[species]
            for channel in channels_of_species:
                self.fish_boxes[channel].extend(bboxes_list)

    def __len__(self):
        return len(self.channel_names)
    
    def __getitem__(self, channel):
        if channel not in self.channel_names:
            return None
        else:
            return self.fish_images[channel]

    def plot(self, channel):
        fig, ax  = plt.subplots(nrows=1, ncols=1)
        ax.imshow(self.fish_images[channel], cmap='gray')
        if channel in self.fish_boxes.keys():
            for box in self.fish_boxes[channel]:
                ax.add_patch(Rectangle(*box, linewidth=1, edgecolor='r', facecolor='none'))
        ax.set_title(f"{channel} Image")
        plt.show()

    def get_number_fluor_channels(self):
        pass

    def get_fluor_channels(self):
        pass

    def get_number_species(self):
        pass

    def plot_predicted_species(self):
        fig, ax = plt.subplots(nrows=1, ncols=5)
        for i, channel in enumerate(self.channel_names, 0):
            ax[i].imshow(self.fish_images[channel], cmap='gray')
            ax[i].set(title=channel)
        
        ax[4].imshow(self.predicted_species_map)
        ax[4].set(title='Species Prediction')
        plt.show(block=False)

    def plot_all_channels(self):
        fig, ax = plt.subplots(nrows=1, ncols=len(self.channel_names))
        for i, channel in enumerate(self.channel_names, 0):
            ax[i].imshow(self.fish_images[channel], cmap='gray')
            if len(self.fish_boxes['channel']) != 0:
                for box in self.fish_boxes[channel]:
                    ax[i].add_patch(Rectangle(*box, linewidth=1, edgecolor='r', facecolor='none'))
            ax[i].set(title=channel)
        plt.show(block=False)