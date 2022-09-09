import glob
import torch
import numpy as np
from collections import OrderedDict
from skimage.measure import regionprops, label
from skimage.io import imread
from skimage.transform import resize, rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit
from torch.utils.data import Dataset, DataLoader
import pathlib
import fastremap
import random
import pickle
from collections import OrderedDict
from torchvision.ops.boxes import box_iou
from narsil2.tracking.networks import trackerNet

np.seterr(divide='ignore', invalid='ignore', over='ignore')
def exp_growth_fit(x, a, b):
	return a * np.exp(-b * x)

colorMap = {'Klebsiella': 'r', 'E.coli' : 'b', 'Pseudomonas': 'g', 'E.cocci' : 'c'}


class trackSingleChannel:
    
    def __init__(self, data_stack_path, tracking_parameters = None, 
                    net_path = None, fileformat='*.tiff', train_mode=False,
                    fish_data = None, species_map = None, flip=False, 
                    fish_data_type = 'single', clean_links=True):
        
        if type(data_stack_path) == pathlib.PosixPath:
            self.data_stack_path = data_stack_path
        elif type(data_stack_path) == str:
            self.data_stack_path = pathlib.Path(data_stack_path)
        
        self.tracking_parameters = tracking_parameters
        self.clean_links = clean_links
        self.flip = flip
        
        if type(net_path) == pathlib.PosixPath:
            self.net_path = net_path
        elif type(net_path) == str:
            self.net_path = pathlib.Path(net_path)
        self.fileformat = fileformat
        self.train_mode = train_mode
        self.true_links = None
        if self.train_mode:
            # set the true links file
            with open(data_stack_path / 'links.pickle', 'rb') as f:
                self.true_links = pickle.load(f)
        # read the stack of images from the directory
        
        self.filenames_list = list(self.data_stack_path.glob(fileformat))
        self.filenames_list = sorted(self.filenames_list, key= lambda x: int(x.stem))
        
        self.n_images = len(self.filenames_list)
        self.images = []
        self.properties = []
        self.n_blobs_per_image = []

        self.bboxes_list = []
        
        for i, filename in enumerate(self.filenames_list):
            image = imread(filename).astype('int')

            if self.flip:
                image = np.flipud(image)
            
            # re-label objects as they will not be in the 
            # same order.
            image, _ = fastremap.renumber(image, in_place=True)
            props = regionprops(image)
            self.n_blobs_per_image.append(len(props))
            self.images.append(image)
            self.properties.append(props)
            bboxes = [torch.tensor(one_blob_props['bbox']) for one_blob_props in props]
            bboxes_stack = torch.stack(bboxes)
            self.bboxes_list.append(bboxes_stack)

        self.time_points = self.n_images - 1
        
        
        self.predicted_links = None
        
        # tracks are dictionary of dictionaries to keep track of objects
        
        self.links_array = None
        self.tracks = []
        
        self.done_track_construction = False
        
        self.fish_data = fish_data
        self.fish_data_type = fish_data_type # used mostly for plotting
        self.species_map = species_map
        self.species_in_channel = None
        
    def _init_net(self):
        net = trackerNet()
        net.load_state_dict(torch.load(self.net_path))
        net.eval()
        return net
    
    def _construct_node_features(self, props):
        row, column, height, width = props['bbox'][0], props['bbox'][1], \
                                    props['bbox'][2] - props['bbox'][0], props['bbox'][3] - props['bbox'][1]
        area = props['area']
        eccentricity = props['eccentricity']
        return [row, column, height, width, area, eccentricity]
    
    def do_tracking(self, in_place=True, edge_feature_size=64):
        # initializie graph/node_data structure
        # read the images/regionprops and construct 
        # properties of the nodes
        graph_data = []  # append dicts containing data from each time step
        for i in range(0, self.time_points):
            data_point = {}
            data_point['node_t'] = len(self.properties[i])
            data_point['node_t1'] = len(self.properties[i + 1])
            num_nodes = len(self.properties[i]) + len(self.properties[i + 1])
            node_features_t = np.asarray([self._construct_node_features(properties) 
                                         for properties in self.properties[i]])
            node_features_t1 = np.asarray([self._construct_node_features(properties)
                                          for properties in self.properties[i + 1]])
            
            node_features_stacked = np.vstack((node_features_t, node_features_t1))
            
            data_point['x'] = torch.from_numpy(node_features_stacked.astype('float32'))
            
            edge_index = [[], []]
            for l in range(data_point['node_t']):
                for m in range(data_point['node_t1']):
                    edge_index[0].append(l)
                    edge_index[1].append(m + data_point['node_t'])
            
            edge_index = np.asarray(edge_index)
            data_point['edge_index'] = torch.from_numpy(edge_index)
            
            # not needed but 
            num_edges = edge_index.shape[1]
            edge_attr = torch.zeros((num_edges, edge_feature_size))
            
            data_point['edge_attr'] = edge_attr
            
            graph_data.append(data_point)
        
        # net loaded from file and set to eval mode
        net = self._init_net()
        
        # run the loop on the graph
        predicted_links = []
        affinity_scores = []

        ious_list = []

        with torch.no_grad():
            for i in range(len(graph_data)):
                x, affinity_matrix_scores = net(graph_data[i])
                # generate links, using argmax and threshold
                links_pred_step = torch.argmax(torch.softmax(affinity_matrix_scores, dim=2), dim=-1)
                predicted_links.append(links_pred_step.numpy())
                affinity_scores.append(affinity_matrix_scores)

                # do  the iou calculations
                ious_timepoint = box_iou(self.bboxes_list[i], self.bboxes_list[i+1])
                ious_timepoint = ious_timepoint.numpy()
                ious_list.append(ious_timepoint)
                
        # do this for all the links that you found between frames
        self.predicted_links = predicted_links
        self.affinity_scores = affinity_scores
        self.ious_list = ious_list 

        if self.clean_links:
            self._clean_raw_links()
        #print("Predicted links: -----")
        #print(self.predicted_links)
        #print("-------------------")
        
        if len(predicted_links) != 0:
            links_all_steps = predicted_links
        elif self.true_links is not None:
            links_all_steps = self.true_links
    
        
        links_array = []
        for i, frame_links in enumerate(links_all_steps, 0):
            # grab all non-zero links [1, 2] in this case
            n_objects_1 = len(self.properties[i])
            n_objects_2 = len(self.properties[i+1])
                
            row, column = np.nonzero(frame_links)
            link_values = frame_links[row, column]
            for l in range(len(link_values)):
                if row[l] < n_objects_1 and column[l] < n_objects_2:
                    links_array.append([i, i + 1, row[l], column[l], link_values[l]])
        if in_place:
            self.links_array = np.asarray(links_array)
        else:
            return np.asarray(links_array)
    
    def _clean_raw_links(self):
        # function to clean the raw links
        # make a local copy and them change the original at the end
        for i in range(self.time_points):
            predicted_links_timestep = self.predicted_links[i].copy()
            iou_timestep = self.ious_list[i]
            
            # clean the movement links
            row, column = np.where(predicted_links_timestep == 1)
            for element in zip(row, column):
                if iou_timestep[element] < self.tracking_parameters['move_iou_thres']:
                    predicted_links_timestep[element] = 0
            
            # clean the divisio links
            row, column = np.where(predicted_links_timestep == 2)
            for element in zip(row, column):
                if iou_timestep[element] < self.tracking_parameters['div_iou_thres']:
                    predicted_links_timestep[element] = 0
            
            self.predicted_links[i] = predicted_links_timestep    
    
    def print_tracks(self):
        if self.done_track_construction:
            for track in self.tracks:
                print(track)
                print("***************")
        else:
            print("Track construction not done")
    
    def create_empty_track(self):
        init_track = {
            'dir' : self.data_stack_path,
            'track_dict': OrderedDict(), # file number, blob number
            'daughters_list': [],
            '_parent_index': [],
            'species': None,
            'fluor_channels': None,
            'areas' : {},
            'lengths': {},
            'time_points': self.time_points,
            'areas_np': -1 * np.ones((self.n_images,)),
            'growth_np': -1 * np.ones((self.n_images,)),
            'areas_rolling_np': -1 * np.ones((self.n_images,)),
            'growth_rolling_np' : -1 * np.ones((self.n_images,)),
            'track_length': 0,
            '_index_to_daughters': []
        }
        
        return init_track
    
    def construct_tracks_from_links(self, daughter_max_area_ratio=0.7, growth_area_ratio_max=0.7):
        if self.links_array is None:
            links_array = self.do_tracking(in_place=False)
        else:
            links_array = self.links_array
        
        links_array = links_array.astype('int')
        # do the loop and construct tracks
        while links_array.size != 0:

            first_row = links_array[0]
            one_track = self.create_empty_track()
            one_track['track_dict'][first_row[0]] = first_row[2]
            one_track['areas'][first_row[0]] = self.properties[first_row[0]][first_row[2]]['area']
            one_track['lengths'][first_row[0]] = self.properties[first_row[0]][first_row[2]]['major_axis_length']
            one_track['track_length'] += 1

            # Now loop till there are no possible connecting objects to the first object
            while True:
                current_blob_index = np.where((links_array[:, 0:4:2] == [first_row[0], first_row[2]]).all(axis=1))[0]

                connections_from_current_blob = np.where((links_array[:, 0:4:2] == [first_row[0], first_row[2]]).all(axis=1))[0]

                if (len(connections_from_current_blob) == 2):
                    # if there are two connections to the currenct blob, set the daughters depending
                    # on the type of connection and stop the tracks and remove the current blob
        
                    possible_daughter_index_1 = connections_from_current_blob[0]
                    possible_daughter_index_2 = connections_from_current_blob[1]

                    if links_array[possible_daughter_index_1][-1] == 2: # 2 is the index for daughter type split
                        # add the daughter numbers to the track data
                        one_track['daughters_list'].append({links_array[possible_daughter_index_1][1] :
                                                             links_array[possible_daughter_index_1][3]})
                        
                    if links_array[possible_daughter_index_2][-1] == 2: # 2 is the index for daughter type split
                        # add the daughter numbers to the track data
                        one_track['daughters_list'].append({links_array[possible_daughter_index_2][1] :
                                                            links_array[possible_daughter_index_2][3]})

                    # delete the connections from current blob
                    links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                    break
                elif (len(connections_from_current_blob) == 1):
                    # only one blob is connected to the current blob
                    # check if some conditions are met and break or 
                    # set the loop to iterate from the next blob after adding the 
                    # connected blob to the current track
                    possible_connection_index = connections_from_current_blob[0]
                    if links_array[possible_connection_index][-1] == 1:
                        # Just a normal movement type of link
                        # check for area increase thresholds and then add it if the blob passes
                        area_ratio = self.properties[first_row[1]][first_row[3]]['area'] / self.properties[first_row[0]][first_row[2]]['area']
                        # growth can only be in a ceratin range, can't increase drastically
                        if area_ratio > growth_area_ratio_max and area_ratio < (1.0/growth_area_ratio_max):
                            # add the blob the track and set the loop to next iteration
                            one_track['track_dict'][first_row[1]] = first_row[3]
                            one_track['areas'][first_row[1]] = self.properties[first_row[1]][first_row[3]]['area']
                            one_track['lengths'][first_row[1]] = self.properties[first_row[1]][first_row[3]]['major_axis_length']
                            one_track['track_length'] += 1
                            #print("Adding single blob", first_row[1], first_row[3])
                            # delete the current connection
                            links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                            # set the loop to begin at the next item connected to the current link
                            next_blob_index = np.where((links_array[:, 0:4:2] == [first_row[1], first_row[3]]).all(axis=1))[0]

                            if next_blob_index.size != 0:
                                first_row = links_array[next_blob_index[0]]

                        else:
                            # doesn't meet the area requirements for adding a movement connection
                            # delete the link and break
                            links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                            break

                    elif links_array[possible_connection_index][-1] == 2:
                        # Could be just linking one daughter incase two daughters weren't found
                        area_ratio = self.properties[first_row[1]][first_row[3]]['area'] / self.properties[first_row[0]][first_row[2]]['area']

                        if area_ratio < daughter_max_area_ratio and area_ratio > (1.0 - daughter_max_area_ratio):
                            one_track['daughters_list'].append({links_array[possible_connection_index][1] :
                                                                links_array[possible_connection_index][3]})
                            # delete the link and break
                            links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                            break
                        else:
                            # doesn't meet daughter break requirements so break and not add the link
                            links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                            break   
                elif (len(connections_from_current_blob) == 0):
                    break
                
                elif (len(connections_from_current_blob) > 2):
                    # delete the link and move on to the next # no track  for this behaviour
                    links_array = np.delete(links_array, (connections_from_current_blob), axis=0)
                    break


            #print(one_track)
            #print("---------------")
            self.tracks.append(one_track)
            
        self.done_track_construction = True
        self.set_daughter_indices()
    

    def set_daughter_indices(self):
        # loop over tracks and set the _index_to_daughters field in all the tracks
        # this is used in looping over tracks easily and jumping between different 
        # tracks that are connected
        for i, track in enumerate(self.tracks, 0):
            if len(track['daughters_list']) > 0:
                # grab all the indices and loop over all the tracks and check
                daughter_file_index = [list(item.keys())[0] for item in track['daughters_list']][0]
                daughter_blob_indices = [list(item.values())[0] for item in track['daughters_list']]
                
                # loop over all the tracks and find the daughter track and modify its properties
                for j, track2 in enumerate(self.tracks, 0):
                    if daughter_file_index in track2['track_dict'] and track2['track_dict'][daughter_file_index] in daughter_blob_indices:
                        track['_index_to_daughters'].append(j)
                        track2['_parent_index'].append(i)
                        
    def label_tracks_with_fluor_channels(self, max_iterations=10):
        if self.fish_data == None:
            print("Fish linking falied")
            return None
        
        # last file key is the same as the time points
        last_key = self.time_points
        for track in self.tracks:
            # check if track ends in the lastFrame
            if last_key in track['track_dict']:
                # label the track with correct fluor channels to the end of the track
                # TODO: should replace this with some IOU like metric, instead of centroid
                centroid = self.properties[last_key][track['track_dict'][last_key]]['centroid']
                track['fluor_channels'] = self.get_fluor_channels(centroid)
        
        #for track in self.tracks:
        #    print(track['fluor_channels'])
        #print("-------")
        
        # Daughter indices are already set in track construction,
        # so loop over a few times (max_iterations) and flow the
        # labels backwards
        while max_iterations > 0:
            for _, track in enumerate(self.tracks, 0):
                indices = track['_index_to_daughters']
                if track['fluor_channels'] == None and len(indices) > 0:
                    # copy channels of the daughters to the parent
                    for index in indices:
                        track['fluor_channels'] = self.tracks[index]['fluor_channels']
                        if track['fluor_channels'] != None:
                            break # from the inner loop
                            
                parent_index = track['_parent_index']
                if track['fluor_channels'] == None and len(parent_index) == 1:
                    track['fluor_channels'] = self.tracks[parent_index[0]]['fluor_channels']
                    
            max_iterations -= 1
        
        #for track in self.tracks:
        #    print(track['fluor_channels'])
            
        # Loop over and check if the track's and their daughter species match
        # if not label it as conflict
        while max_iterations > 0:
            for _, track in enumerate(self.tracks, 0):
                daughter_indices = track['_index_to_daughters']
                if len(daughter_indices) == 2:
                    # if the two daughters don't have and check for None
                    daughter1_channels = self.tracks[daughter_indices[0]]['fluor_channels']
                    daughter2_channels = self.tracks[daughter_indices[1]]['fluor_channels']
                    if ((daughter1_channels is not None) and (daughter2_channels is not None)
                                and (daughter1_channels != daughter2_channels)):
                        track['fluor_channels'] = set(['Conflict'])
                    elif ((daughter1_channels is None) and (daughter2_channels is not None) 
                          and (daughter2_channels != track['fluor_channels'])):
                        track['fluor_channels'] = set(['Confict'])
                    elif ((daughter1_channels is not None) and (daughter2_channels is None)
                         and (daughter1_channels != track['fluor_channels'])):
                        track['fluor_channels'] = set(['Conflict'])
            max_iterations -= 1

    # take a centroid and loops over the bboxes in the fish data
    # and returns a list of fish channels
    def get_fluor_channels(self, centroid):
        
        fluor_channels = []
        for channel in self.fish_data.channel_names:
            if self.inside_box(self.fish_data.fish_boxes[channel], centroid):
                fluor_channels.append(channel)
        
        if len(fluor_channels) == 0:
            return None
        else:
            return set(fluor_channels)
    
    def inside_box(self, boxes, centroid):
        # Checking only x coordinate and forgetting about y coordinate of the boxes for now.
        # TODO: check y corrdinate later # might be useful when you have errors in chip making
        for box in boxes:
            if centroid[0] >= box[0][1] and centroid[0] <= box[0][1] + box[2]:
                return True

        return False
                
    def set_species_for_all_tracks(self):
        
        species_in_channel = set()
        for i, track in enumerate(self.tracks, 0):
            if track['fluor_channels'] is not None:
                for species, fluor_channels in self.species_map.items():
                    #print(species, fluor_channels)
                    #print(track['fluor_channels'])
                    if set(fluor_channels) == set(track['fluor_channels']):
                        track['species'] = species
                        #print("Species added")
                        species_in_channel.add(species)
                        break
        self.species_in_channel = species_in_channel
        
        #for track in self.tracks:
        #    print(track['species'])
    
    def calculate_ratio_growth_track(self, track, plot=False):
        sorted_areas_dict = sorted(track['areas'].items(), key = lambda kv: int(kv[0]))
        start_time = int(sorted_areas_dict[0][0])
        end_time = int(sorted_areas_dict[-1][0])
        time = start_time
        for file_index, areas in sorted_areas_dict:
            track['areas_np'][time] = areas
            time += 1
        
        for i in range(start_time+1, end_time+1):
            track['growth_np'][i] = (track['areas_np'][i] - track['areas_np'][i-1])/ track['areas_np'][i-1]
        
        if plot:
            pass
    
    def calculate_ratio_growth_all_tracks(self):
        for i, track in enumerate(self.tracks):
            self.calculate_ratio_growth_track(track)
    
    def calculate_rolling_growth_track(self, track, width=5, fit_atleast=3, plot=False):
        sorted_areas_dict = sorted(track['areas'].items(), key = lambda kv: int(kv[0]))
        start_time = int(sorted_areas_dict[0][0])
        end_time = int(sorted_areas_dict[-1][0])
        time = start_time
        for file_index, areas in sorted_areas_dict:
            track['areas_rolling_np'][time] = areas
            time += 1
            
        for i in range(width-1, len(track['areas_rolling_np'])):
            current_areas = track['areas_rolling_np'][i - width + 1: i + 1]
            current_times = np.arange(i - width + 1, i + 1)
            set_areas = np.sum(current_areas != -1.0)
            
            if set_areas >= fit_atleast:
                valid_areas_indices = np.where(current_areas != -1.0)
                valid_areas = current_areas[valid_areas_indices]
                valid_times = current_times[valid_areas_indices]
                
                popt, _ = curve_fit(exp_growth_fit, valid_times - valid_times[0], valid_areas)
                
                track['growth_rolling_np'][i]  = -1.0 * popt[1]
        if plot:
            pass
            
        
    def calculate_rolling_growth_all_tracks(self):
        for i, track in enumerate(self.tracks):
            self.calculate_rolling_growth_track(track)
            
    def get_growth_species_labelled_tracks(self, growth_type='rolling'):
        # growth type can be rolling or perecent-ratio change
        growth_rates = {}
        for species in self.species_map:
            growth_rates[species] = []
        
        for i, track in enumerate(self.tracks, 0):
            if track['species'] is not None:
                if growth_type == 'rolling':
                    growth = track['growth_rolling_np']
                elif growth_type == 'ratio':
                    growth = track['growth_np']
                
                species = track['species']
                growth_rates[species].append(growth)
        
        return growth_rates
                
    
    def plot_all_links(self, colors= {1: 'r', 2: 'b', 3: 'm'}, cmap='viridis', interpolation=None):
        if self.predicted_links == None and self.true_links == None:
            return None
        else:
            links = self.predicted_links if self.predicted_links is not None else self.true_links
            height, width = self.images[0].shape
            full_img = np.zeros((height, self.n_images * width))
            for i in range(self.n_images):
                full_img[:, i * width : (i + 1) * width] = self.images[i]
            
            plt.figure()
            plt.imshow(full_img, cmap=cmap, interpolation=interpolation)
            
            for i, frame_links in enumerate(links, 0):
                n_objects_1 = len(self.properties[i])
                n_objects_2 = len(self.properties[i+1])
                
                row, column = np.nonzero(frame_links)
                link_values = frame_links[row, column]
                for l in range(len(link_values)):
                    if row[l] < n_objects_1 and column[l] < n_objects_2:
                        centroid_t_x, centroid_t_y = self.properties[i][row[l]]['centroid']
                        centroid_t1_x, centroid_t1_y = self.properties[i+1][column[l]]['centroid']
                        plt.plot([centroid_t_y + i * (width), centroid_t1_y + (i + 1) * width],
                                [centroid_t_x, centroid_t1_x], colors[link_values[l]])
                        
            plt.title(f"Path: {self.data_stack_path}, ")
            plt.show(block=False)
            
    def plot_all_tracks(self, colors={'move': 'r', 'split': 'g'}):
        pass
    
    def plot_all_links_with_FISH(self, colors={1: 'r', 2: 'g', 3: 'm'}):
        if self.predicted_links == None and self.true_links == None:
            return None
        else:
            links = self.predicted_links if self.predicted_links is not None else self.true_links
            height, width = self.images[0].shape
            full_img = np.zeros((height, self.n_images * width))
            for i in range(self.n_images):
                full_img[:, i * width : (i + 1) * width] = self.images[i]

            # create a matplotlib subplots and put the fish images 
            if self.fish_data is not None:
                fig, ax = plt.subplots(nrows=1, ncols = len(self.fish_data) + 1, 
                            gridspec_kw={'width_ratios': [self.n_images] + [1 for _ in range(len(self.fish_data))]})
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1)
            
            # set the full image and links
            ax[0].imshow(full_img, cmap='gray')
            for i, frame_links in enumerate(links, 0):
                n_objects_1 = len(self.properties[i])
                n_objects_2 = len(self.properties[i+1])
                
                row, column = np.nonzero(frame_links)
                link_values = frame_links[row, column]
                for l in range(len(link_values)):
                    if row[l] < n_objects_1 and column[l] < n_objects_2:
                        centroid_t_x, centroid_t_y = self.properties[i][row[l]]['centroid']
                        centroid_t1_x, centroid_t1_y = self.properties[i+1][column[l]]['centroid']
                        ax[0].plot([centroid_t_y + i * (width), centroid_t1_y + (i + 1) * width],
                                [centroid_t_x, centroid_t1_x], colors[link_values[l]])

            # set all the fish channels and plot their corresponding bboxes

            for i, channel in enumerate(self.fish_data.channel_names, 1):
                ax[i].imshow(self.fish_data[channel], cmap='gray')
                if self.fish_data.transforms == 'box' and len(self.fish_data.fish_boxes[channel]) != 0:
                    for box in self.fish_data.fish_boxes[channel]:
                        ax[i].add_patch(Rectangle(*box, linewidth=1, edgecolor='r', facecolor='none'))
                ax[i].set_title(str(channel))

            plt.show(block=False)

    def plot_all_tracks_with_FISH(self):
        pass

    def plot_all_tracks_for_growth(self):
        pass

    def plot_growth_rates(self):
        pass

    def write_growth_to_file(self):
        
        # create a file name and write the species labelled tracks to a pickle file
        write_dir = self.data_stack_path.parents[1] / self.tracking_parameters['write_dir_names']['growth_rates']
        write_dir.mkdir(parents=True, exist_ok=True)

        filename = self.data_stack_path.stem + '.pickle'
        write_path = write_dir / filename

        growth_rates = self.get_growth_species_labelled_tracks(growth_type=self.tracking_parameters['growth_type'])
        with open(write_path, "wb") as handle:
            pickle.dump(growth_rates, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #print(write_path)
