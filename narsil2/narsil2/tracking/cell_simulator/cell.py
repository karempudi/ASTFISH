# Class of the cells, the growth model and other things related to the cells
# keeping in mind that you have to render and image at the end
# Fully done using numpy
# Adds a lot of randomness

# A lot of the code/ideas comes from https://github.com/modsim/CellSium

import numpy as np
from math import cos, sin, sqrt
from typing import Any, Callable, Dict, Union, Iterator, Tuple, Optional, List
from copy import deepcopy
from narsil2.tracking.cell_simulator.utils import line, shift, circle_segment, parabolic_deformation, rotate

class GenericCell:
    id_counter = 0
    def copy(self):
        return deepcopy(self)
    
    @classmethod
    def next_cell_id(cls):
        cls.id_counter += 1
        return cls.id_counter
    
    @classmethod
    def reset_id(cls):
        print("Resetting ids to start at zero")
        cls.id_counter = 0
        print(cls.id_counter)
    
    def __repr__(self):
        return (
                self.__class__.__name__
            + '('
            + ', '.join(["%s=%r" % (k, v) for k, v in sorted(self.__dict__.items())])
            + ')'
        )
    
    def data_to_write(self):
        data = {}
        data['celltype'] = self.__class__.__name__
        
        for k, v in sorted(self.__dict__.items()):
            data[k] = v
        
        return data


class RectangularCell(GenericCell):
    
    def __init__(self, length=32, width=16, 
                 position=[20, 40], angle=90,
                 birth_time=0, division_size=60,
                 elongation_rate=4,
                 img_size= (1024, 40)):
        # no of rows on the image the cell covers
        self.length = length 
        # no of columns on the image
        self.width = width
        # column x row on the numpy array representing the image
        self.position = position
        self.angle = np.radians(angle)
        
        self.birth_time = birth_time
        self.division_size = division_size
        self.elongation_rate = elongation_rate
        self.lineage_history = []
        self.parent_id = 0
        self.img_size = img_size
    
    def set_id(self, id):
        self.id_ = id
    
    def copy(self):
        copy = super().copy()
        return copy
    
    def birth(self):
        pass
    
    def grow(self, time_step=1):
        self.length += self.elongation_rate * time_step
        
        if self.length > self.division_size:
            offspring_a, offspring_b = self.divide(time_step=time_step)
            offspring_a.length = offspring_b.length = self.length / 2
            
            return (2, (offspring_a, offspring_b))
        # depending on what this returns we add or remove cells
        
        return (1, self)
    
    def divide(self, time_step=1):
        offspring_a, offspring_b = self.copy(), self.copy()
        #offspring_a.id_ = GenericCell.next_cell_id()
        #offspring_b.id_ = GenericCell.next_cell_id()
        
        offspring_a.position, offspring_b.position = self.get_division_positions()
        
        offspring_a.parent_id = offspring_b.parent_id = self.id_
        
        offspring_a.lineage_history = self.lineage_history[:] + [self.id_]
        offspring_b.lineage_history = self.lineage_history[:] + [self.id_]
        
        ## remember birth_times have to be set in the channel state later
        return offspring_a, offspring_b
    
    def step(self, time_step=1):
        #print(f"Cell with id: {self.id_} is stepping {time_step} steps...")
        return self.grow(time_step=time_step)
    
    def get_division_positions(self, count=2):
        
        sin_a, cos_a = sin(self.angle), cos(self.angle)
        
        x, y = self.position
        
        return [
            [float(x + factor * cos_a), float(y + factor * sin_a)]
            for factor in np.linspace(
                -self.length /2 / 2, self.length /2 /2, num=count
            )
        ]
    # rr cc are used for generating images using skimage.draw
    # rr, cc depends on where the cell is in the growth phase
    # be careful with the angle
    
    def raw_points(self, length, width, simplify = False):
        half_width, half_length = width / 2, length / 2

        return np.r_[
            line([+half_length, +half_width],
                 [-half_length, +half_width],
                 num_points=None if not simplify else 3,
            ),
            line([-half_length, +half_width],
                 [-half_length, -half_width],
                 num_points=None if not simplify else 3,
            ),
            line([-half_length, -half_width],
                 [+half_length, -half_width],
                 num_points=None if not simplify else 3,
            ),
            line([+half_length, -half_width],
                 [+half_length, +half_width],
                 num_points=None if not simplify else 3,
            ),
        ]

    def get_points_on_image(self, add_randomness=False, simplify=False):
        
        #if self.length > 3 * self.division_size / 4: 
            # get the coordinates of the two sub imaginary cells and fuse them
        #    current_length = self.length
        #    current_width = self.width
        #    overlap = 0.2 * current_length # randomize this a bit later
        #    points_1 = self.raw_points(current_length/2 + overlap, current_width, simplify=simplify)
        #    points_2 = self.raw_points(current_length/2 + overlap, current_width, simplify=simplify)

        #    position_1 = (self.position[0], self.position[1] - current_length / 4 + overlap /2)
        #    position_2 = (self.position[0], self.position[1] + current_length / 4 - overlap /2)
        #    points_on_image_1 = shift(rotate(points_1, self.angle), position_1)
        #    points_on_image_2 = shift(rotate(points_2, self.angle), position_2)

        #    return (2, (points_on_image_1, points_on_image_2))
        
        points = self.raw_points(self.length, self.width, simplify=simplify)
        points_on_image = shift(rotate(points, self.angle), self.position)
        return (1, points_on_image)


class RodShapedCell(RectangularCell):
    
    def rod_raw_points(self, length, width, simplify = False) :
        diameter = width
        radius = diameter / 2.0
        length = length - diameter
        half_length = length / 2.0

        upper = line(
            [half_length, radius], 
            [-half_length, radius],
            num_points=None if not simplify else 10,
        )
 
        lower = line(
            [-half_length, -radius], 
            [half_length, -radius],
            num_points=None if not simplify else 10,
        )
        circle_left = circle_segment(radius, 90, 270, num_points=None if not simplify else 10)
        circle_left[:, 0] -= half_length

        circle_right = circle_segment(radius, -90, 90, num_points=None if not simplify else 10)
        circle_right[:, 0] += half_length

        return lower,  circle_right, upper, circle_left
        
    def raw_points(self, length, width, simplify=False):
        lower, circle_right, upper, circle_left  = self.rod_raw_points(length, width, simplify=simplify)
        return np.r_[lower, circle_right, upper, circle_left]


class CoccoidCell(RectangularCell):
    
    def __init__(self, length=20, width=20, division_size=30, elongation_rate=2, **kwargs):
        super().__init__(length=length, width=width, division_size=division_size,
                        elongation_rate=elongation_rate, **kwargs)
    
    def raw_points(self, length, width, simplify=False):
        radius = length / 2

        circle_left = circle_segment(radius, 90, 270, num_points=None if not simplify else 5)
        circle_right = circle_segment(radius, -90, 90, num_points=None if not simplify else 5)

        return np.r_[circle_right, circle_left]

class EllipsoidCell(CoccoidCell):
    def __init__(self, length=30, width=20, division_size=40, **kwargs):
        super().__init__(length=length, width=width, division_size=division_size, **kwargs)

    def raw_points(self, length, width, simplify=False):
        points = super().raw_points(length, width, simplify=simplify)
        
        a = self.length / 2
        b = self.width / 2
        
        points[:, 1] *= b/a
        
        return points

class BentRodCell(RodShapedCell):
    
    def __init__(self, bend_overall=0.0, bend_upper=0.0, bend_lower=0.0, **kwargs):
        super().__init__(**kwargs)
        self.bend_overall = bend_overall
        self.bend_upper = bend_upper
        self.bend_lower = bend_lower
    
    def bend(self, points):
        u_idx, l_idx = points[:, 1] > 0, points[:, 1] < 0

        points[u_idx, :] = parabolic_deformation(points[u_idx, :], self.bend_upper)
        points[l_idx, :] = parabolic_deformation(points[l_idx, :], self.bend_lower)

        points = parabolic_deformation(points, self.bend_overall)

        return points
    
    def raw_points(self, length, width, simplify=False):
        lower, circle_right, upper, circle_left = self.rod_raw_points(length, width, simplify=simplify)
        
        points = np.r_[lower, circle_right, upper, circle_left]
        
        points = self.bend(points)
        
        return points


# function that takes in mean, varaince and returns a sample from the 
# distribution, used primarily in adding stochastictiy to the cells
def randomize(mean, parameter_type=None, min_cell_size=12):
    if parameter_type == 'length':
        #return max(mean + abs(int(np.random.normal(loc=0.0, scale=sqrt(mean), size=1))), min_cell_size)
        return max(int(np.random.normal(loc=mean, scale=sqrt(mean), size=1)), min_cell_size)
    elif parameter_type == 'width':
        return None
    elif parameter_type =='division_size':
        #return int(np.random.normal(loc=mean, scale=sqrt(mean), size=1))
        return mean + int(np.random.choice([1, 2, 3]))
        #return mean + abs(int(np.random.normal(loc=0.0, scale=sqrt(mean), size=1)))
    elif parameter_type == 'mean_elongation_rate':
        return max(int(np.random.normal(loc=mean, scale=1.0, size=1)), 0)
        #return max(int(np.random.normal(loc=mean, scale=sqrt(mean), size=1)), 2)
        #return max(mean, mean + int(np.random.choice(np.arange(0, sqrt(mean), 1))))
        #return mean
    elif parameter_type == 'sample_elongation_rate':
        return max(int(np.random.normal(loc=mean, scale=1.0, size=1)), 0)
        #return max(int(np.random.normal(loc=mean, scale=sqrt(mean), size=1)), 2)
        #return max(mean, mean + int(np.random.choice(np.arange(0, sqrt(mean), 1))))
        #return mean

class RectangularCellStochastic(GenericCell):

    def __init__(self, length=32, width=16, position=[20, 40], angle=90, birth_time=0, 
                division_size=60, elongation_rate=4, img_size=(1024, 40), min_cell_size=12):

        # no of rows on the image the cell covers
        self.length = randomize(length, parameter_type='length')
        # no of columns on the image the cell covers
        self.width = width # mean width, randomzie here
        # column x row on th enumpy array representing the image
        self.position = position
        self.angle = np.radians(angle)
        self.min_cell_size = min_cell_size

        self.birth_time = birth_time
        self.division_size = randomize(division_size, parameter_type='division_size') # division size
        self.elongation_rate = randomize(elongation_rate, parameter_type='mean_elongation_rate')# mean elongation rate
        self.lineage_history = []
        self.parent_id = 0
        self.img_size = img_size
        self.previous_elongation_rate_sample = self.elongation_rate


        self.birth()

    def set_id(self, id):
        self.id_ = id
    
    def copy(self):
        copy = super().copy()
        return copy

    def birth(self):
        pass

    def grow(self, time_step=1):
        # current_elongation_rate is sampled from the mean elongation rate (self.elongation_rate)
        # randomize here for growth rate a bit 
        current_elongation_rate = randomize(self.elongation_rate, 'sample_elongation_rate') 
        #print(current_elongation_rate)
        self.length +=  current_elongation_rate * time_step

        if self.length > self.division_size:
            offspring_a, offspring_b = self.divide(time_step=time_step)
            offspring_a.length = offspring_b.length = max(self.length / 2, self.min_cell_size)

            #if offspring_a.length < offspring_a.width:
            #    offspring_a.width = offspring_a.length
            #if offspring_b.length < offspring_b.width:
            #    offspring_b.width = offspring_b.length


            # set a stochastic division size for each offspring
            offspring_a.division_size = randomize(self.length, parameter_type='division_size')
            offspring_b.division_size = randomize(self.length, parameter_type='division_size')

            # set a stochastic mean elongation rate
            offspring_a.elongation_rate = randomize(self.elongation_rate, parameter_type='mean_elongation_rate') 
            offspring_b.elongation_rate = randomize(self.elongation_rate, parameter_type='mean_elongation_rate') 

            return (2 , (offspring_a, offspring_b))
        
        return (1, self)

    def divide(self, time_step=1):
        offspring_a, offspring_b = self.copy(), self.copy()

        offspring_a.position, offspring_b.position = self.get_division_positions()

        offspring_a.parent_id =  offspring_b.parent_id = self.id_

        
        offspring_a.lineage_history = self.lineage_history[:] + [self.id_]
        offspring_b.lineage_history = self.lineage_history[:] + [self.id_]
        
        ## remember birth_times have to be set in the channel state later
        return offspring_a, offspring_b
    

    def step(self, time_step=1):
        return self.grow(time_step=time_step)
    
    def get_division_positions(self, count=2):
        
        sin_a, cos_a = sin(self.angle), cos(self.angle)
        
        x, y = self.position
        
        return [
            [float(x + factor * cos_a), float(y + factor * sin_a)]
            for factor in np.linspace(
                -self.length /2 / 2, self.length /2 /2, num=count
            )
        ]
    # rr cc are used for generating images using skimage.draw
    # rr, cc depends on where the cell is in the growth phase
    # be careful with the angle
    
    def raw_points(self, length, width, simplify = False):
        half_width, half_length = width / 2, length / 2

        return np.r_[
            line([+half_length, +half_width],
                 [-half_length, +half_width],
                 num_points=None if not simplify else 3,
            ),
            line([-half_length, +half_width],
                 [-half_length, -half_width],
                 num_points=None if not simplify else 3,
            ),
            line([-half_length, -half_width],
                 [+half_length, -half_width],
                 num_points=None if not simplify else 3,
            ),
            line([+half_length, -half_width],
                 [+half_length, +half_width],
                 num_points=None if not simplify else 3,
            ),
        ]

    def get_points_on_image(self, add_randomness=False, simplify=False):
        
        if self.length > 3 * self.division_size / 4: 
            # get the coordinates of the two sub imaginary cells and fuse them
            current_length = self.length
            current_width = self.width
            overlap = 0.2 * current_length # randomize this a bit later
            points_1 = self.raw_points(current_length/2 + overlap, current_width, simplify=simplify)
            points_2 = self.raw_points(current_length/2 + overlap, current_width, simplify=simplify)

            position_1 = (self.position[0], self.position[1] - current_length / 4 + overlap /2)
            position_2 = (self.position[0], self.position[1] + current_length / 4 - overlap /2)
            points_on_image_1 = shift(rotate(points_1, self.angle), position_1)
            points_on_image_2 = shift(rotate(points_2, self.angle), position_2)

            return (2, (points_on_image_1, points_on_image_2))
        
        else:
            points = self.raw_points(self.length, self.width, simplify=simplify)
            points_on_image = shift(rotate(points, self.angle), self.position)
            return (1, points_on_image)


class RodShapedCellStochastic(RectangularCellStochastic):
    
    def rod_raw_points(self, length, width, simplify = False) :
        diameter = width
        radius = diameter / 2.0
        length = length - diameter
        half_length = length / 2.0

        upper = line(
            [half_length, radius], 
            [-half_length, radius],
            num_points=None if not simplify else 10,
        )
 
        lower = line(
            [-half_length, -radius], 
            [half_length, -radius],
            num_points=None if not simplify else 10,
        )
        circle_left = circle_segment(radius, 90, 270, num_points=None if not simplify else 10)
        circle_left[:, 0] -= half_length

        circle_right = circle_segment(radius, -90, 90, num_points=None if not simplify else 10)
        circle_right[:, 0] += half_length

        return lower,  circle_right, upper, circle_left
        
    def raw_points(self, length, width, simplify=False):
        lower, circle_right, upper, circle_left  = self.rod_raw_points(length, width, simplify=simplify)
        return np.r_[lower, circle_right, upper, circle_left]


class CoccoidCellStochastic(RectangularCellStochastic):
    
    def __init__(self, length=20, width=20, division_size=30, elongation_rate=2, **kwargs):
        super().__init__(length=length, width=width, division_size=division_size,
                        elongation_rate=elongation_rate, **kwargs)
    
    def raw_points(self, length, width, simplify=False):
        radius = length / 2

        circle_left = circle_segment(radius, 90, 270, num_points=None if not simplify else 5)
        circle_right = circle_segment(radius, -90, 90, num_points=None if not simplify else 5)

        return np.r_[circle_right, circle_left]
