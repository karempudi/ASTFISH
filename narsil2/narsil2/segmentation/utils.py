import numpy as np
import scipy
import ncolor
import edt
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf')
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

import tqdm

from scipy import ndimage as ndi
from skimage.io import imread, imsave
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import glob
import pathlib
from skimage.restoration import rolling_ball
from skimage.filters import difference_of_gaussians, gaussian
from skimage.exposure import equalize_hist
from skimage.morphology import area_opening
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.util.dtype import img_as_ubyte
from skimage import img_as_ubyte


# Function to generate weight maps from a binary segmentation file. 
# The binary segmenation where dtype is uint8 with true values == 255
def generateWeights(filename, sigma = 5, w0 = 10):

    img = imread(filename)
    # removing objects and calculating distances to objects needs labelled images
    labeledImg, num = label(img, return_num=True, connectivity=2)
    # remove small objects
    labeledImg = remove_small_objects(labeledImg, min_size=250)
    # unique values == number of blobs
    unique_values = np.unique(labeledImg) 
    num_values = len(unique_values)
    h, w = labeledImg.shape
    # stack keeps distance maps each blob
    stack = np.zeros(shape=(num_values, h, w))
    for i in range(num_values):
        stack[i] = ndi.distance_transform_edt(~(labeledImg == unique_values[i]))
    # sort the distance
    sorted_distance_stack = np.sort(stack, axis=0)
    # d1 and d2 are the shortest and second shortest distances to each object, 
    # sorted_distance_stack[0] is distance to the background. One can ignore it
    distance_sum = sorted_distance_stack[1] + sorted_distance_stack[2]
    squared_distance = distance_sum ** 2/ (2 * (sigma**2))
    weightmap = w0 * np.exp(-squared_distance)*(labeledImg == 0)
    return weightmap

# Used for generating binary fluorescent mask from fluorescence data,
# Tested against Pseudomonas data only
def generateBinaryMaskFromFluor(fluorImgfilename):
    fluorImg = imread(fluorImgfilename).astype('float32')
    background = rolling_ball(fluorImg, radius = 50)
    fluorImg_remback = fluorImg - background
    filtered = difference_of_gaussians(fluorImg_remback, 1, 4)
    fluorImg_scaled_eq = equalize_hist(filtered)
    filtered_gaussian = gaussian(fluorImg_scaled_eq)
    filtered_median = ndi.median_filter(filtered_gaussian, size=5)
    filtered_gaussian  = filtered_gaussian > 0.92
    image_opening = area_opening(filtered_gaussian, area_threshold=150)
    distance = ndi.distance_transform_edt(image_opening)
    coords = peak_local_max(distance,min_distance = 90, footprint=np.ones((7,7)), labels=image_opening)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image_opening, watershed_line=True, connectivity=2)
    return img_as_ubyte(labels > 0)


def diffuse_particles_heat(T, h_coordinates, w_coordinates, h_med, w_med, cell_w, n_iter, omni=False):
    
    for time in range(n_iter):
        
        
        # add one at the central point every iteration
        T[h_med * cell_w + w_med] += 1
        # spread that one along all the rest of the points on the image
        T[h_coordinates * cell_w + w_coordinates] = 1/ 9.0 * ( T[h_coordinates * cell_w + w_coordinates] +
                                     T[(h_coordinates - 1) * cell_w + w_coordinates] + 
                                     T[(h_coordinates + 1) * cell_w + w_coordinates] + 
                                     T[h_coordinates * cell_w + (w_coordinates - 1)] + 
                                     T[h_coordinates * cell_w + (w_coordinates + 1)] + 
                                     T[(h_coordinates - 1) * cell_w + (w_coordinates - 1)] + 
                                     T[(h_coordinates - 1) * cell_w + (w_coordinates + 1)] + 
                                     T[(h_coordinates + 1) * cell_w + (w_coordinates - 1)] +
                                     T[(h_coordinates + 1) * cell_w + (w_coordinates + 1)]
                                                    )
    
    return T

def labels_to_flow_cpu_heat(label_img, dists, device='cpu', omni=False):
    # convert the labeled images, into flows, calculated on device 
    H, W = label_img.shape
    mu = np.zeros((2, H, W), np.float64)
    mu_c = np.zeros((H, W), np.float64)
    
    # find the maximum label index to get the number of cells
    n_cells = label_img.max()
    # Fit parallelepiped that contains each object and
    # return slices to index into the image
    slices = scipy.ndimage.find_objects(label_img)
    pad = 1
    for i, slice_i in enumerate(slices):
        # 
        if slice_i is not None:
            slice_h, slice_w = slice_i 
            # slices can still have other cells so you mask them out and pad
            # now you have one cell binary mask and paded on all sides with 
            # one pixel
            one_cell_mask = np.pad((label_img[slice_h, slice_w] == i+1), pad)
            # get dimensions of the bounding box of the padded mask
            cell_h, cell_w = one_cell_mask.shape
            
            cell_h = np.int32(cell_h)
            cell_w = np.int32(cell_w)
            h_coordinates, w_coordinates = np.nonzero(one_cell_mask)
            h_coordinates = h_coordinates.astype(np.int32)
            w_coordinates = w_coordinates.astype(np.int32)
            
            T = np.zeros(cell_h* cell_w, np.float64)
            # Iterations = 2 * (length + width) of the object
            n_iter = 2 * np.int32(np.ptp(w_coordinates) + np.ptp(h_coordinates))
            
            # median
            h_med, w_med = np.median(h_coordinates), np.median(w_coordinates)
            # calculate the point for some strange reason, idk yet
            # but you might as well use h_med, w_med
            i_min = np.argmin((h_coordinates - h_med)** 2 + (w_coordinates - w_med)**2)
            h_med, w_med = np.array([h_coordinates[i_min]], np.int32), np.array([w_coordinates[i_min]], np.int32)
            
            # for n iterations, start a particle 1.0 intensity at the median 
            # and propagate it to all the neigbouring points by repeated averaging
            # (probably can leverage convolutions, later for this diffusion)
            T = diffuse_particles(T, h_coordinates, w_coordinates, h_med, w_med, cell_w, n_iter, omni=omni)
            
            # approximate the derivative of the the diffusion map
            dy = (T[(h_coordinates + 1) * cell_w + w_coordinates] 
                  - T[(h_coordinates - 1) * cell_w + w_coordinates]) / 2
            dx = (T[h_coordinates * cell_w + (w_coordinates + 1)]  
                 -  T[h_coordinates * cell_w + (w_coordinates - 1)]) / 2
            
            # Now add the diffusion gradients to the image the size of the 
            # the label_img, ie mu
            mu[:, slice_h.start + h_coordinates - pad, slice_w.start + w_coordinates - pad] = np.stack((dy, dx))
            mu_c[slice_h.start + h_coordinates - pad, slice_w.start + w_coordinates - pad] = T[h_coordinates * cell_w + w_coordinates]
    
    # normalize the field, basically divide by the mean for each direction
    # mu is (2 * H * W) shape
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)
    
    # return normalize
    return mu, mu_c


def labels_to_flows(label_img, omni=False, use_gpu=False):
    # convert the label_img containing labels: 1,2,... (n_cells)
    # to a 4 color mask for easier computation of distances, i guess
    label_img = ncolor.format_labels(label_img)
    dists = edt.edt(label_img)
    
    mu, T = labels_to_flow_cpu(label_img, dists, device='cpu', omni=omni)
    
    return label_img, dists, T, mu

def flows_to_labels(flows):
    pass

#############################################################
################### Plotting functions ######################
#############################################################

# Get input data from different dataloaders and plot them,
# Generally use it to plot a batch
def plot_datapoints_deprecated(img_data):
    
    phase_imgs, label_imgs = img_data
    # Type conversion
    if type(phase_imgs) == torch.Tensor:
        phase_imgs = phase_imgs.numpy()
    if type(label_imgs) == torch.Tensor:
        label_imgs = label_imgs.numpy()

    if phase_imgs.ndim == 4:
        pass
    elif phase_imgs.ndim == 3:
        phase_imgs = phase_imgs[np.newaxis, ]
    elif phase_imgs.ndim == 2:
        phase_imgs = phase_imgs[np.newaxis, np.newaxis, ]
    
    if label_imgs.ndim == 4:
        # got a batch, so do nothing
        pass
    elif label_imgs.ndim == 3:
        # add new axis as you received a stack of labels
        label_imgs = label_imgs[np.newaxis, ]
    elif label_imgs.ndim == 2:
        # pad dimension if you got only one image
        label_imgs = label_imgs[np.newaxis, np.newaxis, ]

    batch_size, phase_channels, H, W = phase_imgs.shape
    _, labels_channels, _, _ = label_imgs.shape

    n_imgs_per_figure = phase_channels + labels_channels  + 2 + 1 + 1 

    for batch_idx in range(batch_size):
        
        # if there is only one channels, calculate flows
        if labels_channels == 1:
            label_img, dists, T, mu = labels_to_flows(label_imgs[batch_idx, 0])

        fig, ax = plt.subplots(1, n_imgs_per_figure)
        for i in range(n_imgs_per_figure):
            if i < phase_channels:
                ax[i].imshow(phase_imgs[batch_idx, i, :, :], cmap='gray')
            elif i >= phase_channels and i < phase_channels + labels_channels:
                ax[i].imshow(label_imgs[batch_idx, i-phase_channels, :, :])
            elif i >= phase_channels + labels_channels and i < phase_channels + labels_channels + 2:
                # flows
                ax[i].imshow(mu[i-phase_channels - labels_channels])
            elif i >= phase_channels + labels_channels + 2 and i < phase_channels + labels_channels + 3:
                # distance
                ax[i].imshow(dists)
            else:
                ax[i].imshow(T)

        fig.suptitle("One single datapoint")
        plt.show()

#
def plot_datapoints(img_data):
    # just plot all the stuff there is 
    phase_imgs, label_imgs = img_data
    # Type conversion
    if type(phase_imgs) == torch.Tensor:
        phase_imgs = phase_imgs.numpy()
    if type(label_imgs) == torch.Tensor:
        label_imgs = label_imgs.numpy()

    if phase_imgs.ndim == 4:
        pass
    elif phase_imgs.ndim == 3:
        phase_imgs = phase_imgs[np.newaxis, ]
    elif phase_imgs.ndim == 2:
        phase_imgs = phase_imgs[np.newaxis, np.newaxis, ]
    
    if label_imgs.ndim == 4:
        # got a batch, so do nothing
        pass
    elif label_imgs.ndim == 3:
        # add new axis as you received a stack of labels
        label_imgs = label_imgs[np.newaxis, ]
    elif label_imgs.ndim == 2:
        # pad dimension if you got only one image
        label_imgs = label_imgs[np.newaxis, np.newaxis, ]

    batch_size, phase_channels, H, W = phase_imgs.shape
    _, labels_channels, _, _ = label_imgs.shape
    n_imgs_per_figure = phase_channels + labels_channels
    for img_idx in range(batch_size):
        fig, ax = plt.subplots(1, n_imgs_per_figure)
        for i in range(n_imgs_per_figure):
            if i < phase_channels:
                ax[i].imshow(phase_imgs[img_idx, i, :, :], cmap='gray')
            else:
                ax[i].imshow(label_imgs[img_idx, i-phase_channels, :, :])
        plt.show()

# Calculate and write flows for fast loading of the data
#
def write_flows(mask_dir, write_dir, fileformat='*.png'):
    if not type(mask_dir) == Path:
        mask_dir = Path(mask_dir)
    if not type(write_dir) == Path:
        write_dir = Path(write_dir)

    img_names = list(mask_dir.glob(fileformat))

    # if write_dir is not there, create it 
    if not write_dir.exists():
        write_dir.mkdir(parents=True, exist_ok=True)

    flow_dir_y = write_dir / "flow_y"
    flow_dir_y.mkdir(parents=True, exist_ok=True)
    flow_dir_x = write_dir / "flow_x"
    flow_dir_x.mkdir(parents=True, exist_ok=True)

    for i, img_name in enumerate(img_names, 0):
        mask_img = io.imread(img_names[i])
        mask_img, dists, T, mu = labels_to_flows(mask_img)

        file_name = img_name.name

        io.imsave(flow_dir_y / file_name, mu[0])
        io.imsave(flow_dir_x / file_name, mu[1])
        print(f"Flows for {file_name} written ...")

    print(f"Flows calculated for {len(img_names)} files ... :)")

