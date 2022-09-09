import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from narsil2.tracking.cell_simulator.utils import get_image
from skimage import io, img_as_ubyte, draw
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import pickle
import json

class ChannelState:
    """ Class that is used to simulate the cells in channels"""
    
    def __init__(self, initial_cells, time = 0,
                img_size = (1024, 40), top_boundary = 120, bottom_boundary = 680,
                 distance_bn_cells = 2
                ):
        # clean ids
        
        self.cells = []
        self.cells_to_add = []
        self.cells_to_remove = []
        self.time = time
        self.img_size = img_size
        self.top_boundary = top_boundary
        self.bottom_boundary = bottom_boundary
        self.distance_bn_cells = distance_bn_cells
        
        self.cell_id_counter = 0
        # will hold images
        self.images = []
        # add initial cells
        for i, cell in enumerate(initial_cells, 1):
            cell.set_id(i)
            self.cells_to_add.append(cell)
        self.cell_id_counter += len(initial_cells)
        
        self.commit()
        self.update_positions()
        self.update_images()
        

        
        # just the links as a numpy array of shape n_objects_t1 x n_objects_t2
        # codes for state changes
        # move = 1
        # division = 2
        # leaving = 3
        # appearance = 4
        # death = 5, we will ignore death for now
        self.links = []
        
    def clear(self):
        self.cells.clear()
        self.cells_to_add.clear()
        self.cells_to_remove.clear()
        self.time = 0
        
    def add(self, cell):
        self.cells_to_add.append(cell)
    
    def remove(self, cell):
        self.cells_to_remove.append(cell)
        
    def commit(self):
        for cell in self.cells_to_add:
            self.cells.append(cell)
        
        for cell in self.cells_to_remove:
            self.cells.remove(cell)
        self.cells_to_remove.clear()
        self.cells_to_add.clear()
    
    def copy(self):
        new_channel_state = self.__class__()
        new_channel_state.cells = self.cells[:]
        new_channel_state.cells_to_add = self.cells_to_add[:]
        new_channel_state.cells_to_remove = self.cells_to_remove[:]
        
        return new_channel_state
    
    def step(self, time_step=1):
        self.time += time_step
        
        # print before
        #print("-------------------------")
        #for cell in self.cells:
        #    print(cell)
        #print(f"Time now: {self.time}")
        
        n_cells_before = len(self.cells)
        cell_ids_before = []
        for cell in self.cells:
            cell_ids_before.append(cell.id_)
            
        division_ids = {}
        just_moved_ids = []
        for cell_number, cell in enumerate(self.cells, 0):
            
            returned_cells = cell.step(time_step=time_step)
            if returned_cells[0] == 2:
                # changing offspring birth times
                returned_cells[1][0].birth_time = self.time
                self.cell_id_counter += 1
                returned_cells[1][0].set_id(self.cell_id_counter)
                returned_cells[1][1].birth_time = self.time
                self.cell_id_counter += 1
                returned_cells[1][1].set_id(self.cell_id_counter)
                
                self.add(returned_cells[1][0])
                self.add(returned_cells[1][1])
                
                division_ids[cell.id_] = [returned_cells[1][0].id_, returned_cells[1][1].id_]
                
                # remove the mother if there was split
                self.remove(cell)
            
            # cell just grew and didn't divide
            elif returned_cells[0] == 1:
                just_moved_ids.append(cell.id_)
                
        self.commit()
        
        # update positions
        self.update_positions()
        
        n_cells_after = len(self.cells)
        cell_ids_after = []
        for cell in self.cells:
            cell_ids_after.append(cell.id_)
        
        
        links_per_step = np.zeros((n_cells_before, n_cells_after))
        for i in range(n_cells_before):
            if cell_ids_before[i] in division_ids:
                # cell divided find the number in next frame
                daughter_1_index = cell_ids_after.index(division_ids[cell_ids_before[i]][0])
                daughter_2_index = cell_ids_after.index(division_ids[cell_ids_before[i]][1])
                links_per_step[i, daughter_1_index] = 2
                links_per_step[i, daughter_2_index] = 2
            else:
                moved_cell_index = cell_ids_after.index(cell_ids_before[i])
                links_per_step[i, moved_cell_index] = 1
                
        # remove cells out of the boundary
        #print(f"Cells before: {cell_ids_before}")
        #print(f"Cells after: {cell_ids_after}")
        removed_cell_ids = []
        for cell in self.cells:
            if cell.position[1] > self.bottom_boundary:
                removed_cell_ids.append(cell.id_)
                self.remove(cell)
        #print(removed_cell_ids)
        #print("-----------")
        # find the columns corresponding to the removed_ids
        # and set the element that is 1 (from the linking before)
        # and set it to 3 (code for leaving)
        for cell_id in removed_cell_ids:
            removed_index = cell_ids_after.index(cell_id)
            #print("Removed index:", removed_index)
            index_to_change = np.where(
                            np.logical_or(links_per_step[:, removed_index] == 1,
                                          links_per_step[:, removed_index] == 2))[0][0]
            links_per_step[index_to_change, removed_index] = 3
        
        # commit changes after removing the cells that have
        # go beyond boudary
        
        self.update_links(links_per_step)
        
        self.commit()
        
        # update images 
        self.update_images()
        
        # print cells after
        #for cell in self.cells:
        #    print(cell)
        #print("------------------------")

    def update_positions(self):
    
        self.cells.sort(key= lambda cell: cell.position[1])
        
        current_top = self.top_boundary
        for cell in self.cells:
            changed_position = current_top + cell.length / 2.0
            cell.position[1] = changed_position
            current_top = changed_position + cell.length / 2.0 + self.distance_bn_cells
    
    def update_links(self, links_array):
        self.links.append(links_array)
        
    def update_images(self):
        new_img = np.zeros(self.img_size)
        for cell in self.cells:
            vertices = cell.get_points_on_image(simplify=False)
            if vertices[0] == 1:
                rr, cc = draw.polygon(vertices[1][:, 1], vertices[1][:, 0], self.img_size)
                new_img[rr, cc] = cell.id_
            elif vertices[0] == 2:
                rr1, cc1 = draw.polygon(vertices[1][0][:, 1], vertices[1][0][:, 0], self.img_size)
                rr2, cc2 = draw.polygon(vertices[1][1][:, 1], vertices[1][1][:, 0], self.img_size)
                new_img[rr1, cc1] = cell.id_
                new_img[rr2, cc2] = cell.id_
        
        self.images.append(new_img)
            
    def save_state(self, path):
        with open(path, 'wb') as handle:
            pickle.dump(self.links, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def save_images(self, path):
        data_to_write = np.asarray(self.images)
        np.save(path, data_to_write, allow_pickle=True)
    
    def get_stack(self, time_points=15, plot=True, colors = {1: 'r', 2: 'g', 3: 'm'}):
        for i in range(time_points):
            self.step()


        # calculate regionprops to link centroids for the links, 
        # color of the link depends on the type of link

        if plot:    
            num_images = len(self.images)
            height, width = self.img_size
            full_img = np.zeros((height, num_images * width))
            properties = []
            for i in range(num_images):
                full_img[:, i*width:(i+1)*width] = (self.images[i] > 0)
                properties.append(regionprops(label(self.images[i])))
                
            plt.figure()
            plt.imshow(full_img, cmap='gray')
            # plot links with different colors
            for i, frame_links in enumerate(self.links, 0):
                n_objects_1 = len(properties[i])
                n_objects_2 = len(properties[i+1])
                # loop over nonzero links and plot
                row, column = np.nonzero(frame_links)
                link_values = frame_links[row, column]
                for l in range(len(link_values)):
                    if row[l] < n_objects_1 and column[l] < n_objects_2:
                        centroid_t_x, centroid_t_y = properties[i][row[l]]['centroid']
                        centroid_t1_x, centroid_t1_y = properties[i+1][column[l]]['centroid']
                        plt.plot([centroid_t_y + i * (width), centroid_t1_y + (i + 1) * (width)], 
                                [centroid_t_x, centroid_t1_x], colors[link_values[l]])
            plt.show()
            plt.title(f"Timepoints : {time_points+1}, including 0")

        return np.asarray(self.images).astype('uint16') , self.links