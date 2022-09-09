from functools import lru_cache
import numpy as np
from typing import Optional, Tuple, List
from skimage.draw import polygon


@lru_cache(maxsize=128)
def cached_linspace(start: float, stop: float, num: int) -> np.ndarray:
    array = np.linspace(start=start, stop=stop, num=num)
    array.setflags(write=False)
    return array

def line(start: np.ndarray,
        stop: np.ndarray,
        interval: float = 0.1,
        minimum_points: int = 10,
        num_points: Optional[int] = None) -> np.ndarray:
    
    start, stop = np.atleast_2d(start), np.atleast_2d(stop)
    delta  = stop - start

    # create a point every interal,
    # we will get bavk to units to use in the simulation later
    if num_points is None:
        num_points = max(int(np.linalg.norm(delta)/ interval) + 1, minimum_points)
    
    ramp = cached_linspace(start=0.0, stop=1.0, num=num_points)
    ramp = np.c_[ramp, ramp]
    
    return start + delta * ramp

def circle_segment(radius: float,
                    start: np.ndarray,
                    stop: np.ndarray,
                    interval: float = 0.1,
                    minimum_points: int = 10,
                    num_points: Optional[int] = None) -> np.ndarray:

    interval = np.arctan(float(interval) / radius)
    start, stop = np.radians(start), np.radians(stop)

    if num_points is None:
        num_points = max(int((stop - start) / interval), minimum_points)
    
    ramp = cached_linspace(start, stop, num_points)

    return radius * np.c_[np.cos(ramp), np.sin(ramp)]


def parabolic_deformation(array: np.ndarray, factor: float) -> np.ndarray:
    array[:, 1] += factor * (array[:, 0] ** 2 - (array[:, 0] ** 2).max())
    return array

def get_rotation_matrix(angle: float) -> np.ndarray:
    
    return np.array([[np.cos(angle), -np.sin(angle)], 
                     [np.sin(angle), np.cos(angle)]])
            

def rotate(data: np.ndarray, angle: float) -> np.ndarray:
    return np.dot(np.atleast_2d(data), get_rotation_matrix(angle).T)

def shift(data: np.ndarray, vector: np.ndarray) -> np.ndarray:
    
    data = data.copy()
    vector = np.atleast_2d(vector)

    for n in range(min(data.shape[1], vector.shape[1])):
        data[:, n] += vector[0, n]
    
    return data

def get_image(cells: List[object], size: Tuple[int, int]) -> np.ndarray:

    img = np.zeros(size, 'uint8')

    for cell in cells:
        vertices = cell.points_on_image(simplify=True)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], img.shape)
        img[rr,cc] = cell.id_

    return img 