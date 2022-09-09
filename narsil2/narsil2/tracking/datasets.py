from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import io
from torchvision import transforms
import numpy as np
import torch
import fastremap
import pickle
from pprint import pprint
from torch_geometric.data import Data as geom_Data
from skimage.measure import label, regionprops
from narsil2.tracking.cell_simulator.cell import RodShapedCell, CoccoidCell, RodShapedCellStochastic
from narsil2.tracking.cell_simulator.channel import ChannelState

class channelStackBipartite(geom_Data):
    
    def __init__(self, time_points=1, current_window=1, hidden_size=128):
        super().__init__()
        self.time_points = time_points
        self.current_window = current_window
        self.hidden_size = hidden_size
        
        self.data = {}
        self._generate_data()

        
        # keys are time frames, values are list of nodes
        # these are copies
        # The convention is always the source nodes are detections
        # The target nodes are associations
        self.source_nodes = {} # 0 to t (included, total t+1 keys)
        self.target_nodes = {} # 1 to t (included, total t keys)
        self.edges_s2t = {}
        self.edges_t2s = {}
        self.edge_weights ={}
        
        self._init_graph()
        
    def _generate_data(self):
        ecoli_cell = RodShapedCell()
        channel = ChannelState([ecoli_cell])
        images, links = channel.get_stack(time_points=self.time_points, plot=False)
        # here is where you can apply augmentations, 
        # this is the exact place, may generate labels as well here for the nodes
        # or write something to generate targets for ablations here as well if possible
        self.data['images'] = images
        self.data['links'] = links
    
    def _init_graph(self):
        # loop over the time and construct the elements per time step
        # construct lists
        frames = self.__len__() + 1
        # do regionprops on the stack
        regionprops_stack = [regionprops(label(self.data['images'][i])) for i in range(frames)]
        for i in range(frames):
            n_objects = len(regionprops_stack[i])
            # for each blob build props
            self.source_nodes[i] = np.asarray([self._constrct_source_node_features(properties)
                                            for properties in regionprops_stack[i]])
            
        # target nodes' numbers depend on number of objects in frame t-1 and t
        for i in range(1, frames):
            num_target_nodes = len(regionprops_stack[i-1]) * len(regionprops_stack[i])
            #print("Number of target_nodes:", num_target_nodes)
            self.target_nodes[i] = np.zeros((num_target_nodes, self.hidden_size))
            
    def _constrct_source_node_features(self, props):
        row, column, height, width = props['bbox'][0], props['bbox'][1], \
                                    props['bbox'][2] - props['bbox'][0], props['bbox'][3] - props['bbox'][1]
        area = props['area']
        eccentricity = props['eccentricity']
        return [row, column, height, width, area, eccentricity]
    
    def __len__(self):
        return len(self.data['links'])
    
    # get one running window of the graph from the graph datastructure
    # to get the latest features
    def __getitem__(self, idx):
        # helps iterating over the time series using a current_window
        if isinstance(idx, int):
            # target node features can vary in size if they are updated before or just raw properties
            x_source = []
            # add the idx timepoint nodes' features, these could be computed hidden states
            x_source.append(self.source_nodes[idx])
            # add the idx+1 nodes' features
            x_source.append(self.source_nodes[idx+1])
            # target nodes are always going to have the same shape, due to zero initialization
            x_target = self.target_nodes[idx+1]

            n_objects_1 = x_source[0].shape[0]
            n_objects_2 = x_source[1].shape[0]
            n_target_nodes = x_target.shape[0]
            # t2s might be easier to construct 
            edges_t2s = []
            labels = { 'source': np.zeros((n_objects_1 + n_objects_2, ), dtype='int'),
                       'target': np.zeros((n_target_nodes, ), dtype='int')
                     }
            for i in range(n_target_nodes):
                edges_t2s.append([i, i%n_objects_1])   
                edges_t2s.append([i, n_objects_1 + i//n_objects_1])

                link_from = i%n_objects_1
                link_to = i // n_objects_1
                # you can't do the below two lines here as they overwrite the source labels unfortunately
                #labels['source'][i%n_objects_1] = self.data['links'][idx][link_from, link_to]
                #labels['source'][n_objects_1 + i//n_objects_1] = self.data['links'][idx][link_from, link_to]

                # target labels describe the transitions that are undergo from previous timepoint
                # to the current timepoint
                labels['target'][i] = self.data['links'][idx][link_from, link_to]

            # source labels should describe is something drastic happened to the cell
            # 0 means cell is fine, all cells in t-1 frame are set to 0 by default
            # 1 means it divided from the previous step, anomalous growth like some segmentation
            # error or merging behaviour between cells
            # all cells in frame t are set to how they came into existence
            for i in range(n_objects_1):
                labels['source'][i] = 0

            frame1_idx, frame2_idx = np.nonzero(self.data['links'][idx])

            num_links = len(frame2_idx)
            for j in range(num_links):
                if frame2_idx[j] < n_objects_2:
                    # if the link is greater than 1, means that the cells divided or left or something happened
                    labels['source'][n_objects_1 + frame2_idx[j]] = int(self.data['links'][idx][frame1_idx[j], frame2_idx[j]] > 1)

            edges_t2s = np.asarray(edges_t2s).T.copy()

            # edges_s2t is just flipping rows in edge_t2s

            edges_s2t = np.array([edges_t2s[1], edges_t2s[0]]).copy()

            # edgeweights to do the subtraction of detection/source node features
            edge_weight_s2t = np.asarray([[-1.0, +1.0] * n_target_nodes]).astype('float32')


            return {
                'images': torch.from_numpy(self.data['images'][idx: idx+2].astype('float16')),
                'links': torch.from_numpy(self.data['links'][idx]),
                'x_s': [torch.from_numpy(x.astype('float32')) for x in x_source],
                'x_t': torch.from_numpy(x_target.astype('float32')),
                'edges_s2t': torch.from_numpy(edges_s2t),
                'edges_t2s': torch.from_numpy(edges_t2s),
                'edge_weight_s2t': torch.from_numpy(edge_weight_s2t),
                'labels': {
                    'source': torch.from_numpy(labels['source']),
                    'target': torch.from_numpy(labels['target'])
                }
            }
        elif isinstance(idx, slice):
            # if it is a slice, get the slices and stitch them up correctly
            start = idx.start
            stop = idx.stop
            step = idx.step
            print(f"Slice start: {start}, stop: {stop}, step: {step}")
            
            graph = {}
            
            # add the first timestep
            #
            
            # iterate and add the other timesteps
            #for i in range(start, stop):
            #    first_timestep = self.__getitem__(i)
            #    second_time
            
            return None
        
    
    def plot_graph(self, idx):
        pass
    
    def plot_images(self):
        
        num_images, height, width = self.data['images'].shape
        full_img = np.zeros((height, num_images * width))
        for i in range(num_images):
            full_img[:, i*width:(i+1)*width] = self.data['images'][i]
        
        plt.figure()
        plt.imshow(full_img)
        plt.title(f"Timepoints : {self.time_points+1} (starting at t=0)")
        plt.show()


class channelStackGraph(geom_Data):
    
    def __init__(self, time_points=1, edge_features_size=64):
        super().__init__()
        self.time_points = time_points
        self.edge_features_size = edge_features_size
        
        self.data = {}
        self._generate_data()
        
        self.nodes = {} # one key for each time step, 0 to t-1, for t timesteps 
        self.edges = {} # one key for each time step, 0 to t-1 for t timesteps
        self.edge_attributes = {}
        # for each timestep 
        self.edge_labels = {} 
        
        self._init_graph()
    
    def _generate_data(self):
        ecoli_cell = RodShapedCell()
        #coccoid_cell = CoccoidCell()
        channel = ChannelState([ecoli_cell])
        images, links = channel.get_stack(time_points=self.time_points, plot=False)
        
        self.data['images']  = images
        self.data['links'] = links
    
    def __len__(self):
        return len(self.data['links'])
    
    def _construct_node_features(self, props):
        row, column, height, width = props['bbox'][0], props['bbox'][1], \
                                    props['bbox'][2] - props['bbox'][0], props['bbox'][3] - props['bbox'][1]
        area = props['area']
        eccentricity = props['eccentricity']
        return [row, column, height, width, area, eccentricity]
    
    def _init_graph(self):
        frames = self.__len__() + 1
        regionprops_stack = [regionprops(label(self.data['images'][i])) for i in range(frames)]
        
        # interate and build a graph
        
        for i in range(0, frames - 1):
            num_nodes = len(regionprops_stack[i]) + len(regionprops_stack[i+1])
            pooled_properties = regionprops_stack[i] + regionprops_stack[i+1]
            self.nodes[i] = []
            self.nodes[i].append(np.asarray([self._construct_node_features(properties) 
                                        for properties in regionprops_stack[i]]))
            self.nodes[i].append(np.asarray([self._construct_node_features(properties)
                                            for properties in regionprops_stack[i+1]]))
    
    def __getitem__(self, idx):
        
        node_features = self.nodes[idx]
        n_objects_1 = node_features[0].shape[0]
        n_objects_2 = node_features[1].shape[0]
        # create edge_index
        edge_index = [[], []]
        # basically add connections between nodes at t and t+1
        for i in range(n_objects_1):
            for j in range(n_objects_2):
                edge_index[0].append(i)
                edge_index[1].append(j+n_objects_1)
        
        #edge_index_repeat = [edge_index[0] + edge_index[1], edge_index[1] + edge_index[0]].copy()
        
        #edge_index_repeat = np.asarray(edge_index_repeat)
        edge_index_repeat = np.asarray(edge_index)
        # create edge_attribute
        num_edges = edge_index_repeat.shape[1]
        edge_attr = np.zeros((num_edges, self.edge_features_size))
        
        # create labels for the edge attributes to generate the affinity matrix
        
        x = np.vstack((node_features[0], node_features[1]))
        
        return {
            'images': self.data['images'][idx:idx+2].astype('float16'),
            'links': torch.from_numpy(self.data['links'][idx][:n_objects_1, :n_objects_2].astype('int')),
            'x': torch.from_numpy(x.astype('float32')),
            'edge_index': torch.from_numpy(edge_index_repeat),
            'edge_attr': torch.from_numpy(edge_attr),
            'node_t': n_objects_1,
            'node_t1': n_objects_2
        }
    
    def plot_images(self, cmap='viridis'):
        
        num_images, height, width = self.data['images'].shape
        full_img = np.zeros((height, num_images * width))
        for i in range(num_images):
            full_img[:, i*width:(i+1)*width] = self.data['images'][i]
        
        plt.figure(figsize=(10, 6))
        plt.imshow(full_img, cmap=cmap)
        plt.title(f"Timepoints : {self.time_points+1} (starting at t=0)")
        plt.show()

class channelStackGraphFaster(geom_Data):
    
    def __init__(self, time_points=1, edge_features_size=64,
                images_dirpath=None, links_dirpath=None, datapoint_number=None):
        super().__init__()
        self.time_points = time_points
        self.edge_features_size = edge_features_size
        if type(images_dirpath) == str:
            self.images_dirpath = Path(images_dirpath)
        if type(links_dirpath) == str:   
            self.links_dirpath = Path(links_dirpath)
            
        self.datapoint_number = datapoint_number
        
        self.data = {}
        if datapoint_number is None:
            self._generate_data()
        else:
            self._load_data_from_file()
        
        self.nodes = {} # one key for each time step, 0 to t-1, for t timesteps 
        self.edges = {} # one key for each time step, 0 to t-1 for t timesteps
        self.edge_attributes = {}
        # for each timestep 
        self.edge_labels = {} 
        
        self._init_graph()
    
    def _generate_data(self):
        ecoli_cell = RodShapedCellStochastic(
                                length=46, width=30, position=[40, 40],
                            division_size=84, elongation_rate=4,
                        img_size=(2048, 80))
        #coccoid_cell = CoccoidCell()
        channel = ChannelState([ecoli_cell], img_size=(2048, 80), top_boundary=240, bottom_boundary=1600)
        images, links = channel.get_stack(time_points=self.time_points, plot=False)
        
        self.data['images']  = images
        self.data['links'] = links
        
    def _load_data_from_file(self):
        
        # read from file_path provided and load them appropriately 
        links_filename = str(self.datapoint_number) + '.pickle'
        links_filepath = self.links_dirpath / links_filename
        with open(links_filepath, 'rb') as f:
            links = pickle.load(f)
            
        # loop over images and accumulate
        images_dirname = self.images_dirpath / str(self.datapoint_number)
        images_filepaths = sorted(list(images_dirname.glob('*.tiff')), key= lambda name: int(name.stem))
        
        images_list = []
        number_images = len(images_filepaths)
        for i in range(number_images):
            image = io.imread(images_filepaths[i])
            images_list.append(image)
        
        self.time_points = number_images - 1
        self.data['images'] = np.asarray(images_list)
        self.data['links'] = links
    
    def __len__(self):
        return len(self.data['links'])
    
    def _construct_node_features(self, props):
        row, column, height, width = props['bbox'][0], props['bbox'][1], \
                                    props['bbox'][2] - props['bbox'][0], props['bbox'][3] - props['bbox'][1]
        area = props['area']
        eccentricity = props['eccentricity']
        return [row, column, height, width, area, eccentricity]
    
    def _init_graph(self):
        frames = self.__len__() + 1
        regionprops_stack = [regionprops(label(self.data['images'][i])) for i in range(frames)]
        
        # interate and build a graph
        
        for i in range(0, frames - 1):
            num_nodes = len(regionprops_stack[i]) + len(regionprops_stack[i+1])
            pooled_properties = regionprops_stack[i] + regionprops_stack[i+1]
            self.nodes[i] = []
            self.nodes[i].append(np.asarray([self._construct_node_features(properties) 
                                        for properties in regionprops_stack[i]]))
            self.nodes[i].append(np.asarray([self._construct_node_features(properties)
                                            for properties in regionprops_stack[i+1]]))
    
    def __getitem__(self, idx):
        
        node_features = self.nodes[idx]
        n_objects_1 = node_features[0].shape[0]
        n_objects_2 = node_features[1].shape[0]
        # create edge_index
        edge_index = [[], []]
        # basically add connections between nodes at t and t+1
        for i in range(n_objects_1):
            for j in range(n_objects_2):
                edge_index[0].append(i)
                edge_index[1].append(j+n_objects_1)
        
        #edge_index_repeat = [edge_index[0] + edge_index[1], edge_index[1] + edge_index[0]].copy()
        
        #edge_index_repeat = np.asarray(edge_index_repeat)
        edge_index_repeat = np.asarray(edge_index)
        # create edge_attribute
        num_edges = edge_index_repeat.shape[1]
        edge_attr = np.zeros((num_edges, self.edge_features_size))
        
        # create labels for the edge attributes to generate the affinity matrix
        
        x = np.vstack((node_features[0], node_features[1]))
        
        return {
            'images': self.data['images'][idx:idx+2],
            'links': torch.from_numpy(self.data['links'][idx][:n_objects_1, :n_objects_2].astype('int')),
            'x': torch.from_numpy(x.astype('float32')),
            'edge_index': torch.from_numpy(edge_index_repeat),
            'edge_attr': torch.from_numpy(edge_attr),
            'node_t': n_objects_1,
            'node_t1': n_objects_2
        }
    
    def plot_images(self, cmap='viridis', colors = {1: 'r', 2: 'g', 3: 'm'}, interpolation=None):
        
        num_images, height, width = self.data['images'].shape
        full_img = np.zeros((height, num_images * width))
        properties = []
        for i in range(num_images):
            image = self.data['images'][i]
            full_img[:, i*width:(i+1)*width] = image
            image, _ = fastremap.renumber(image, in_place=True)
            properties.append(regionprops(image))
            
        links = self.data['links']
        
        plt.figure(figsize=(10, 6))
        plt.imshow(full_img, cmap=cmap, interpolation=interpolation)
        
        for i, frame_links in enumerate(links, 0):
            n_objects_1 = len(properties[i])
            n_objects_2 = len(properties[i+1])
            
            row, column = np.nonzero(frame_links)
            link_values = frame_links[row, column]
            for l in range(len(link_values)):
                if row[l] < n_objects_1 and column[l] < n_objects_2:
                    centroid_t_x, centroid_t_y = properties[i][row[l]]['centroid']
                    centroid_t1_x, centroid_t1_y = properties[i+1][column[l]]['centroid']
                    plt.plot([centroid_t_y + i * (width) , centroid_t1_y + (i + 1) * width],
                            [centroid_t_x , centroid_t1_x], colors[link_values[l]])
                

        plt.title(f"Timepoints : {self.time_points+1} (starting at t=0)")
        plt.show()