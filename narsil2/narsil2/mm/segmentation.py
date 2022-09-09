# code useful for segmenting a mother machine experiment
# Cell segmentation can be done in 2 ways (normal U-net and Omnipose net) 
# Channel segmentation is done using only U-net, smaller version
from scipy import ndimage
import torch
import os
import time
import sys
import numpy as np
import multiprocessing as mp
from narsil2.segmentation.networks import basicUnet, smallerUnet
from narsil2.segmentation.lightning_modules import UTrainingModule
from narsil2.segmentation.datasets import phaseTestDir, MMDataTest
from narsil2.segmentation.transformations import resizeOneImage, tensorizeOneImage
from narsil2.segmentation.utils_omni import reconstruct_masks_cpu_omni
from narsil2.fish.transformations import FishTransformations
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage.morphology import binary_dilation, binary_erosion, binary_fill_holes, binary_opening
from scipy.signal import find_peaks
from skimage.io import imread, imsave
from skimage.transform import resize, rotate
from skimage.exposure import equalize_adapthist
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import label, find_contours
from collections import OrderedDict
from functools import partial
from pathlib import Path

###########################################################
############# Segmentation functions ######################
###########################################################


def segment_cells_dir(position_phase_dir, model, segmentation_parameters,
    device="cpu"):
    """
    # Segments a directory of images with all the same shape using data loader
    # Use only if they are all the same shape
    # Model should already be loaded with parameters, Use loadNet function to load the model
    # from file to the correct model.
    Arguments:
    ----------
    position_phase_dir: Path
        Path of the directory containing phase contrast images that are to
        be segmented, string usually ends in position name ..../Pos101/ 
    model: 
        torch model for cell segmentation
    
    segmentation_parameters: dict
        A dictionary containinig all the parameters needed for segmentation

    Returns
    -------

    Succcess/Failure: bool
        Success if everything goes well in this directory
    """
    phase_image_transform = segmentation_parameters['cell_transforms']

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    if type(position_phase_dir) == str:
        position_phase_dir = Path(position_phase_dir)

    position_str = position_phase_dir.parents[0].stem
    
    # create empty directory in the analysis dir with appropriate position number
    analysis_dir = Path(segmentation_parameters['analysis_dir']) / position_str / segmentation_parameters['write_dir_names']['cell_seg_raw']
    analysis_dir.mkdir(parents=True, exist_ok=True)


    segmentation_type = segmentation_parameters['seg_method']['cells']
    #sys.stdout.write(f"Segmenting cells with type: {segmentation_type}\n")
    #sys.stdout.flush()
    if segmentation_type == "unet":
        phase_data_loader = None # TODO: implement later
    elif segmentation_type == "omnipose":
        phase_dataset = MMDataTest(dataset_path=position_phase_dir,
                                 phase_fileformat=segmentation_parameters['fileformat'],
                                 mothermachine_data=True, 
                                 transforms=phase_image_transform)
        phase_data_loader = DataLoader(phase_dataset, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        for i_batch, data_test in enumerate(phase_data_loader, 0):
            phase_test = data_test['phase'].to(torch_device)
            #print(data_test['filename'])
            filename_to_save = Path(data_test['filename'][0]).name
            _, _, H, W =  phase_test.shape
            if segmentation_type == 'omnipose' and (W > 2048):
                # split image into two parts and then do segmentation and stitch back
                # might not be ideal, might want to cut it somewhere where there
                # are no cells, but it is fine for AST kinda experiments
                out1 = model(phase_test[:, :, :, :2048])
                out2 = model(phase_test[:, :, :, 2048:])

                # stitch the outputs
                out = torch.cat((out1, out2), dim=3)
                #print(out.shape)
            elif segmentation_type == "omnipose" and (W < 2048):
                out = model(phase_test)
            elif segmentation_type == "unet":
                pass
            
            # reconstruction step
            if segmentation_type == 'omnipose':
                generated_mask = reconstruct_masks_cpu_omni(out.cpu().numpy()[0],
                                    cell_prob_threshold=segmentation_parameters['cell_prob_threshold'],
                                    clean_mask=segmentation_parameters['clean_mask'],
                                    min_size=segmentation_parameters['min_size'],
                                    device=device,
                                    fast=True)
                #generated_mask = generated_mask % 255

            
            # imporatant to delete this to free space on GPU and not wait for it to be
            # garbage collected. Things break otherwise
            del out
                
            # unpad the generated mask to the original shape

            # write the generated mask to disk
            if segmentation_parameters['save_seg_mask']:
                write_path = analysis_dir / filename_to_save 
                imsave(write_path, generated_mask.astype('float32'), plugin='tifffile', compress=6, check_contrast=False)

        sys.stdout.write(f"{position_str} segmenting cells --- Done..\n")
        sys.stdout.flush()
        return 0


def segment_channels_dir(position_phase_dir, model, segmentation_parameters, device='cpu'):
    """
    # Segments a directory of images with all the same shape using data loader
    # Use only if they are all the same shape
    # Model should already be loaded with parameters, Use loadNet function to load the model
    # from file to the correct model.
    Arguments:
    ----------
    positionPhaseDir: Path
        Path of the directory containing phase contrast images that are to
        be segmented, string usually ends in position name ..../Pos101/ 
    model: 
        torch model for channel segmentation
    
    segmentationParameters: dict
        A dictionary containinig all the parameters needed for segmentation

    Returns
    -------

    Succcess/Failure: bool
        Success if everything goes well in this directory

    positionImgChop: dict
        A dictionary with keynames as imagefilenames and values are numpy arrays of the 
        locations where you can cut out the individual channels from the images
    """
    phase_image_transform = segmentation_parameters['channel_transforms']

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")


    if type(position_phase_dir) == str:
        position_phase_dir = Path(position_phase_dir)

    position_str = position_phase_dir.parents[0].stem

    channel_pos_filename = Path(segmentation_parameters['analysis_dir']) / position_str / 'channel_locations.npy'

    create_dir_name = channel_pos_filename.parents[0]
    create_dir_name.mkdir(parents=True, exist_ok=True)

    analysis_dir = Path(segmentation_parameters['analysis_dir']) / position_str / segmentation_parameters['write_dir_names']['channel_seg_raw']
    analysis_dir.mkdir(parents=True, exist_ok=True)


    segmentation_type = segmentation_parameters['seg_method']['channels']
    #sys.stdout.write(f"Segmneting channels with type: {segmentation_type}\n")
    #sys.stdout.flush()
    if segmentation_type == "unet":
        phase_dataset = phaseTestDir(position_phase_dir, transform=phase_image_transform) 

        phase_data_loader = DataLoader(phase_dataset, batch_size=1, shuffle=False, num_workers=4) 
    elif segmentation_type == "omnipose":
        phase_dataset = MMDataTest(dataset_path=position_phase_dir,
                                 phase_fileformat=segmentation_parameters['fileformat'],
                                 mothermachine_data=True, 
                                 transforms=phase_image_transform)
        phase_data_loader = DataLoader(phase_dataset, batch_size=1, shuffle=False, num_workers=4)

    channel_cutting_failed = False
    position_channel_locations = {}
    with torch.no_grad():
        for i_batch, data_test in enumerate(phase_data_loader, 0):
            phase_test = data_test['phase'].to(torch_device)
            #print(data_test['filename'])
            filename_to_save = Path(data_test['filename'][0]).name
            #print(phase_test.shape)
            _, _, H, W =  phase_test.shape

            if segmentation_type == 'unet' and channel_cutting_failed == False:
                channel_pred = torch.sigmoid(model(phase_test)) > segmentation_parameters['channel_seg_threshold']
                channel_pred = channel_pred.to("cpu").detach().numpy().squeeze(0).squeeze(0)

                # try getting the locations
                try:
                    position_channel_locations[filename_to_save] = get_channel_locations_single_image(channel_pred,
                                                                segmentation_parameters['channel_cutting_params']
                                                                )
                except Exception as e:
                    channel_cutting_failed = True
                    print(e)
                    print(f"{position_str} has failed cutting channels")
                    continue

            if channel_cutting_failed == True:
                position_channel_locations[filename_to_save] = np.asarray([])

            if segmentation_parameters['save_channel_seg_mask']:
                write_path = analysis_dir / filename_to_save 
                imsave(write_path, channel_pred.astype('float32'), plugin='tifffile', compress=6, check_contrast=False)

            #sys.stdout.write(f"{filename_to_save} channels -- Done\n")
            #sys.stdout.flush()

        # for every position write the channel locations in the appropriate directory
        np.save(channel_pos_filename, position_channel_locations)
        sys.stdout.write(f"{position_str} -- channel seg done...\n")
        sys.stdout.flush()
        #return (True, position_channel_locations)
    
    #return (False, position_channel_locations)

# Function to find channel locations to cut from the segmented channel image and some parameters
def get_channel_locations_single_image(channel_pred, params):
    hist = np.sum(channel_pred[params['channel_min']: params['channel_max']], axis=0) > params['channel_sum']

    peaks, _ = find_peaks(hist, distance=params['hist_peaks_distance'])

    indices_with_larger_gaps = np.where(np.ediff1d(peaks) > params['minimum_barcode_distance'])[0]

    possible_barcode_locations = indices_with_larger_gaps[np.argmax(indices_with_larger_gaps > params['first_barcode_index'])]

    num_channels = params['num_channels']
    before_barcode_locations = np.zeros((num_channels,), dtype=int)
    after_barcode_locations = np.zeros((num_channels,), dtype=int)

    for i in range(num_channels):
        before_barcode_locations[i] = peaks[possible_barcode_locations - i]
        after_barcode_locations[i] = peaks[possible_barcode_locations + i + 1]

    locations_to_cut = np.concatenate((before_barcode_locations[::-1], after_barcode_locations), axis=0)

    return locations_to_cut


def load_net(model_path, segmentation_type='omnipose', device="cpu"):
    """
    A function that takes in a model file and returns a net in eval mode 
    Arguments
    ---------
    model_path: .pth model file
        A model file, .pth file containing the model details to load the correct 
        model file
    segmentation_type: (str)
        Segmentation type can be 'omnipose' or 'unet', omnipose model loads from 
        checkpoint (.ckpt) files and unet model loads from .pth file
    device: str 
        usually "cuda:0" or "cuda:1"
    Returns
    -------
    net: torch.nn.Module object
        Net in eval mode
    """

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    if segmentation_type == 'omnipose':
        # Load net from the the checkpoint file
        net = UTrainingModule.load_from_checkpoint(model_path)
        net.to(device)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        return net

    elif segmentation_type == 'unet':
        # Read the model file
        saved_model = torch.load(model_path)

        # use the net depending on what model is loaded
        if saved_model['modelParameters']['netType'] == 'big':
            net = basicUnet(saved_model['modelParameters']['transposeConv'])
        elif saved_model['modelParameters']['netType'] == 'small':
            net = smallerUnet(saved_model['modelParameters']['transposeConv'])

        # load the net
        net.load_state_dict(saved_model['model_state_dict'])

        # send to device
        net.to(device)

        # eval mode
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

        return net



# Function for looping over multiple positions and segmenting them continuously
def segment_cells_all_positions(phase_main_dir, model, positions, 
            segmentation_parameters, device="cpu"):

    if type(phase_main_dir) == str:
        phase_main_dir = Path(phase_main_dir)

    for position in positions:
        dir_string = 'Pos' + str(position)
        position_phase_dir = phase_main_dir / dir_string / segmentation_parameters['dir_names']['phase']
        # do the segmentation and writing the files for one position
        segment_cells_dir(position_phase_dir, model, segmentation_parameters,
            segmentation_parameters['segmentation_devices']['cells'])

# Function for looping over mulitple positions and segmenting them continuously
def segment_channels_all_positions(phase_main_dir, model, positions, 
            segmentation_parameters, device="cpu"):

    if type(phase_main_dir) == str:
        phase_main_dir = Path(phase_main_dir)
    for position in positions:
        dir_string = 'Pos' + str(position)
        position_phase_dir = phase_main_dir / dir_string / segmentation_parameters['dir_names']['phase'] 
        # do the segmenting and channel cutting operations for one position
        segment_channels_dir(position_phase_dir, model, segmentation_parameters,
            segmentation_parameters['segmentation_devices']['channels'])


#def segment_all_positions(phase_main_dir, positions, models, segmentation_parameters):
def segment_all_positions(phase_main_dir, segmentation_parameters, positions):
    """ 
    Function to segment position ranges given to the function.
    Networks are loaded on respective devices specified in the segmentation_parameters

    # Two processes are launched that will run segmentation of cells and channels
    separetly and independantly at the same time and write results to disk

    Arguments
    ---------
    phase_main_dir: str or pathlib.PosixPath
        Path of the directory where positions with phase images live
    positions: range object or list 
        Range or list of positions in the directroy or the ones you want to 
        segment
    
    models: dict
        A dict with paths of different models (for segmentation of cells and 
            channels). "cells" and "channels" are keys and modelPaths are values
    segmentation_parameters: dict
        Dict containing all the parameters needed for segmentation like GPU devices,
        image resize and shrink components, segmentation threshold, etc
    
    Returns
    -------
    None
    """
    start = time.time()

    cells_device = segmentation_parameters['segmentation_devices']['cells']

    channels_device = segmentation_parameters['segmentation_devices']['channels']

    # load the cell segmentation network and on the correct device
    cell_net = load_net(segmentation_parameters['model_paths']['cells'],
                    segmentation_type=segmentation_parameters['seg_method']['cells'],
                    device=cells_device,
                    )
    print("Segmentation Network loaded successfuly ...")

    # load the cell segmentation network and on the correct device
    channels_net = load_net(segmentation_parameters['model_paths']['channels'],
                    segmentation_type=segmentation_parameters['seg_method']['channels'],
                    device=channels_device
                    )
    print("Channel Segmentation Network loaded successfully ... ")

    analysis_dir = Path(segmentation_parameters['analysis_dir'])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    print("Created analysis writing directories ... \n")

    #try:
    #    mp.set_start_method('spawn')
    #except RuntimeError:
    #    pass
    ctx = mp.get_context('spawn')

    cell_process = ctx.Process(target=segment_cells_all_positions, 
        args=(phase_main_dir, cell_net, positions, segmentation_parameters,
                segmentation_parameters['segmentation_devices']['cells'],
                        ))

    channel_process = ctx.Process(target=segment_channels_all_positions, 
            args=(phase_main_dir, channels_net, positions, segmentation_parameters,
                segmentation_parameters['segmentation_devices']['channels']))

    cell_process.start()
    channel_process.start()

    channel_process.join()
    cell_process.join()
    duration = time.time() - start
    print(f"Duration for segmentation is {duration}s")

    #

    #cuttingLocations = {}

    #for position in positions:
    #    positionPhaseDir = phaseMainDir + 'Pos' + str(position) + '/'
    #    success, channelCuttingFailed, positionsImgChopMap = segmentPosDirectory(positionPhaseDir, net, channelNet, segmentationParameters)
    #    if (success and (channelCuttingFailed == False)):
    #        print(positionPhaseDir)
    #        cuttingLocations[position] = positionsImgChopMap

#    duration = time.time() - start
#    print(f"Duration for segmenting {positions} is {duration}s")
#
#    print("Writing cutting locations ...")
#    print(cuttingLocations.keys(), " positions have cutting locations done")
#    np.save(segmentationParameters['saveResultsDir'] + 'channelLocations.npy', cuttingLocations)


#########################################################
########### Channel Stack cutting functions #############
#########################################################
def cut_channels_one_position(analysis_position_dir, segmentation_parameters):
    """
    Funciton to write and cut individual mother machine channels and write them for one
    position (which is a stack of images)
    """

    if type(analysis_position_dir) == str:
        analysis_position_dir = Path(analysis_position_dir)

    position_str = analysis_position_dir.stem
    channel_locations_filename = analysis_position_dir / 'channel_locations.npy' 

    # channel locations has keys (image filenames and )
    channel_locations = np.load(channel_locations_filename, allow_pickle=True).item()
    filenames = sorted(channel_locations.keys())

    # directory for writing the files
    blobs_dir = analysis_position_dir / segmentation_parameters['write_dir_names']['channel_stacks']
    blobs_dir.mkdir(parents=True, exist_ok=True)

    # segmented raw data dir
    segmented_dir = analysis_position_dir / segmentation_parameters['write_dir_names']['cell_seg_raw']

    channel_width = segmentation_parameters['cutting_and_writing_params']['channel_width']

    # do the cutting if there are locations for all the images in the directory
    if len(filenames) == segmentation_parameters['cutting_and_writing_params']['num_images']:
        
        for i, filename in enumerate(filenames, 0):
            if (i >= segmentation_parameters['cutting_and_writing_params']['cut_until_frames']):
                sys.stdout.write(f"Cutting segmented channels of {position_str} --- Done.\n")
                sys.stdout.flush()
                return None
            
            img_filename = segmented_dir / filename 
            image = imread(img_filename)

            peaks = channel_locations[filename]
            left = peaks - (channel_width // 2) 
            right = peaks + (channel_width // 2)
            channel_limits = list(zip(left, right))

            for l in range(len(channel_limits)):
                stack_dir  = blobs_dir / str(l)
                stack_dir.mkdir(parents=True, exist_ok=True)
                img_chop = image[:, channel_limits[l][0] : channel_limits[l][1]]
                chop_filename = str(i) + '.tiff'
                imsave(stack_dir / chop_filename, img_chop, plugin='tifffile', 
                        compress=6, check_contrast=False)

    else:
        sys.stdout.write(f"Cutting failed at position : {position_str} \n")
        sys.stdout.flush()
    
    return None


def cut_channels_all_positions(analysis_main_dir, positions, 
                segmentation_parameters, num_processes=6):
    """
    Function to write and cut individual mother machine channels and write
    them to appropriate directories, for cell-tracking and growth rate analysis
    later
    Arguments
    ---------
    analysis_main_dir: str or pathlib.PosixPath
        Path of the analysis directory containing the segmented images directory
    positions: range object or list
        Range or list of position numbers to cut
    segmentation_parameters: dict
        Dictionary containing parameters used for segmentation
    num_processes: int
        Number of processes you want to parallelize on
    Returns
    -------
    None
    """

    start = time.time()

    if type(analysis_main_dir) == str:
        analysis_main_dir = Path(analysis_main_dir)

    list_position_dirs = []
    for position in positions:
        dir_string = 'Pos' + str(position)
        directory = analysis_main_dir / dir_string 
        list_position_dirs.append(directory)
    #print(list_position_dirs)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    pool = mp.Pool(processes=num_processes)
    pool.map(partial(cut_channels_one_position, segmentation_parameters=segmentation_parameters), list_position_dirs)
    pool.close()
    pool.join()

    duration = time.time() - start
    print(f"Duration of cutting channels of {len(list_position_dirs)} positions is {duration}s")

    return None


def cut_fluor_one_position(analysis_position_dir, fluorescence_parameters):

    if type(analysis_position_dir) == str:
        analysis_position_dir = Path(analysis_position_dir)
    
    if type(fluorescence_parameters['genotype_dir']) == str:
        genotype_dir = Path(fluorescence_parameters['genotype_dir'])
    else:
        genotype_dir = fluorescence_parameters['genotype_dir']

    position_str = analysis_position_dir.stem
    #print(position_str)

    write_dir = analysis_position_dir / fluorescence_parameters['write_dir_names']['fluor_stacks']
    write_dir.mkdir(parents=True, exist_ok=True)

    # read the channel locations from the channel locations file
    channel_locations_filename = analysis_position_dir / 'channel_locations.npy'
    channel_locations = np.load(channel_locations_filename, allow_pickle=True).item()
    filenames = sorted(channel_locations.keys())
    if fluorescence_parameters['phase_img_to_map'] in filenames:
        for channel_name in fluorescence_parameters['channel_names']:
            
            # read the images and do some transformations

            fluor_img_filename = genotype_dir / position_str / channel_name / 'img_000000000.tiff'
            #print(fluor_img_filename)
            channel_img  = imread(fluor_img_filename, as_gray=True)

            channel_img = fluorescence_parameters['transform'](channel_img)
            channel_img = channel_img.astype('uint16')
            if fluorescence_parameters['equalize'] == 'equalize_adapthist':
                channel_img = (65535 * equalize_adapthist(channel_img))

            peaks = channel_locations[fluorescence_parameters['phase_img_to_map']]
            left = peaks - (fluorescence_parameters['channel_width'] // 2)
            right = peaks + (fluorescence_parameters['channel_width'] // 2)
            channel_limits = list(zip(left, right))

            for l in range(len(channel_limits)):
                # make dirs and write the chops
                img_chop_write_dir = write_dir / str(l)
                img_chop_write_dir.mkdir(parents=True, exist_ok=True)
                img_chop =  channel_img[:, channel_limits[l][0]: channel_limits[l][1]]
                filename = str(channel_name) + '.tiff'
                imsave(img_chop_write_dir / filename, img_chop, plugin='tifffile',
                         compress=6, check_contrast=False)
        
        sys.stdout.write(f"{position_str} fluor channels cutting done .. \n")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"{position_str} couldn't cut fluor channels .. \n")
        sys.stdout.flush()

    return None

def cut_fluor_all_positions(analysis_main_dir, positions, 
            fluorescence_parameters, num_processes=6):

    start = time.time()

    if type(analysis_main_dir) == str:
        analysis_main_dir = Path(analysis_main_dir)

    list_position_dirs = []
    for position in positions:
        dir_string = 'Pos' + str(position)
        directory = analysis_main_dir / dir_string 
        list_position_dirs.append(directory)
    #print(list_position_dirs)
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    pool = mp.Pool(processes=num_processes)
    pool.map(partial(cut_fluor_one_position, fluorescence_parameters=fluorescence_parameters), list_position_dirs)
    pool.close()
    pool.join()

    duration = time.time() - start
    print(f"Duration of cutting fluor channels of {len(list_position_dirs)} positions is {duration}s")

    return None
