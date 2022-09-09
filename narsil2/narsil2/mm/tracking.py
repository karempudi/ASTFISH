### Tracking functions to analyze positions
import numpy as np
import glob
import os
import sys
import time
import pickle
import multiprocessing as mp
import shutil
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from narsil2.tracking.track import trackSingleChannel
from narsil2.fish.datasets import singleColourFISHData, multiColorFISHData

def process_one_channel(one_channel_path, fish_data, tracking_parameters, flip, fish_data_type):
    
    #print(one_channel_path)
    stack = trackSingleChannel(data_stack_path=one_channel_path,
            net_path = tracking_parameters['net_path'], 
            tracking_parameters=tracking_parameters,
            train_mode=False, fish_data=fish_data,
            species_map=tracking_parameters['species_map'],
            flip=flip, fish_data_type=fish_data_type,
            )
    stack.do_tracking()
    #print(one_channel_path)
    #stack.plot_all_links_with_FISH()
    stack.construct_tracks_from_links()
    stack.label_tracks_with_fluor_channels()
    stack.set_species_for_all_tracks()
    stack.calculate_ratio_growth_all_tracks()
    stack.calculate_rolling_growth_all_tracks()
    stack.write_growth_to_file()


def process_one_position(analysis_position_dir, tracking_parameters, fluor_parameters,
            process_type='single'):

    position_str = analysis_position_dir.stem
    position = int(position_str[3:])

    if process_type == 'single':
        background_channel_no = tracking_parameters['background_channel_no']
        # generate fish background data and 
        #if position in tracking_parameters['flip_positions']:
        bg_dirname = analysis_position_dir / tracking_parameters['write_dir_names']['fluor_stacks'] / str(background_channel_no)
        bg_fish_data = singleColourFISHData(bg_dirname, fluor_parameters['channel_names'])

        for i in range(0, tracking_parameters['num_channels']):
            try:
                one_channel_path = analysis_position_dir / tracking_parameters['write_dir_names']['channel_stacks'] / str(i)
                fish_dirname = analysis_position_dir / tracking_parameters['write_dir_names']['fluor_stacks'] / str(i)
                flip = position in tracking_parameters['flip_positions']
                fish_data = singleColourFISHData(fish_dirname, fluor_parameters['channel_names'],
                                fluor_channel_threshold=fluor_parameters['fluor_channel_thres'],
                                transforms='box', background_fishdata=bg_fish_data,
                                min_box_height=fluor_parameters['min_box_height'],
                                flip=flip)
                #print(fish_data.channel_names)
                process_one_channel(one_channel_path, fish_data, tracking_parameters, flip, fish_data_type=process_type)
                #print(one_channel_path)
            except Exception as e:
                print(e)

        sys.stdout.write(f"{position_str} -- Done\n")
        sys.stdout.flush()
    
    elif process_type == 'double':
        # 2- color FISH takes a slightly different way of assigning species using a
        # a classifier instead of a threshold

        for i in range(0, tracking_parameters['num_channels']):
            try:

                # process on channel, track, assign species and write growth rates
                one_channel_path = analysis_position_dir / tracking_parameters['write_dir_names']['channel_stacks'] / str(i)
                fish_dirname = analysis_position_dir / tracking_parameters['write_dir_names']['fluor_stacks'] / str(i)
                flip = position in tracking_parameters['flip_positions']

                # fish_data will hold all the information 
                fish_data = multiColorFISHData(fish_dirname=fish_dirname,
                            channel_names=fluor_parameters['channel_names'],
                            classifier_filename=fluor_parameters['classifier_path'],
                            bg_normalization_vector=fluor_parameters['normalization_vector'],
                            smooth=fluor_parameters['smooth'],
                            flip=flip)
                process_one_channel(one_channel_path, fish_data, tracking_parameters, flip, fish_data_type=process_type)


            except Exception as e:
                sys.stdout.write(f"{position_str} --- {e}\n")
                sys.stdout.flush()
        
        sys.stdout.write(f"{position_str} -- Done\n")
        sys.stdout.flush()


def process_all_positions(analysis_main_dir, positions, 
        tracking_parameters, fluor_parameters, num_processes=6, process_type='single'):
    """
    Function to process all positions, to construct tracks from the 
    segmented mother-machine data and bundle Fish channels approprietely 
    or just do tracking as per parameters sets
    Arguments
    ---------
    analysis_main_dir: str or pathlib.PosixPath
        Path of the analysis directory containing all the analysis data

    positions: range object or list
        Range or list of position numbers to analyze
    
    tracking_parameters: dict
        Dictionary containing all the parameters need for tracking, species
        -assignment, growth calculations and writing, etc
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
        # check if the blobs dir and fish channels dir are not empty, only the add it to the list
        blobs_dir = directory / tracking_parameters['write_dir_names']['channel_stacks']
        fish_dir = directory / tracking_parameters['write_dir_names']['fluor_stacks']
        if len(os.listdir(blobs_dir)) == 0 or len(os.listdir(fish_dir)) == 0:
            pass
        else: 
            list_position_dirs.append(directory)
    #print(list_position_dirs)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    if process_type == 'single':
        pool = mp.Pool(processes=num_processes)
        pool.map(partial(process_one_position, tracking_parameters=tracking_parameters,
                    fluor_parameters=fluor_parameters, process_type=process_type), list_position_dirs)
        pool.close()
        pool.join()

        duration = time.time() - start
        print(f"Duration of cutting channels of {len(list_position_dirs)} is {duration}s")
    elif process_type == 'double':
        for position_directory in list_position_dirs:
            process_one_position(position_directory, tracking_parameters=tracking_parameters,
                    fluor_parameters=fluor_parameters, process_type=process_type)

        duration = time.time() - start
        print(f"Duration of cutting channels of {len(list_position_dirs)} is {duration}s")
 
    return None

def delete_data():
    pass