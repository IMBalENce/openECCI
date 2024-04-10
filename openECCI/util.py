# Copyright 2023-2024 The openECCI developers
#
# This file is part of openECCI.
#
# openECCI is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# openECCI is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with openECCI. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2
from scipy.optimize import minimize
import pathlib
import h5py as h5
from scipy.ndimage import rotate
from openECCI.ctf import file_reader as ctf_reader
# ctf_reader is currently loaded as a separate plugin, however, 
# it will be integrated in orix.io in the future for easier access.

def normalize(image_array: np.ndarray):
    """
    Normalize an image array into the range of [0, 1] to facilitate image comparison.
    
    Parameters
    ----------
    image_array
        Numpy 2d array of the image
    
    Returns
    -------
        Normalised Numpy 2d array of the image
    """
    norm = np.linalg.norm(image_array)
    image_norm = image_array/norm  # normalized image
    return image_norm

def background_corr(image_array: np.ndarray, sigma):
    """
    Flat field correction to remove low spatial frequency backgroud by using the rolling ball method.
    
    Parameters
    ----------
    image_array
        Numpy 2d array of the image
    sigma
        Rolling ball diameter, or the gaussian blur sigma value
    
    Returns
    -------
        Normalised Numpy 2d array of the image
    """
    blurred = skimage.filters.gaussian(
        normalize(image_array), sigma=(sigma, sigma), truncate=3.5, channel_axis=2)
    flatten = normalize(image_array)/blurred
    return flatten

def enhance_contrast(image_array: np.ndarray, clipLimit=1.0, tileGridSize=8):
    """
    Enhance pattern contrast by using CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm.
    TODO: the input images needs to be 8bit, otherwise the algorithm does not work, fix 8-bit to any-bit
    
    Parameters
    ----------
    filename
        Numpy 2d array of the image
    clipLimit
        
    tileGridSize
        
    
    Returns
    -------
        Enhanced 2d array of the image
    """

    try:
        image = image_array / image_array.max()
    except:
        image = image_array.data / image_array.data.max()
    image = (image * 2 ** 8).astype('uint8')

    tileGridSize = int(tileGridSize)

    clahe = cv2.createCLAHE(clipLimit=clipLimit,
                            tileGridSize=(tileGridSize, tileGridSize))
    image = clahe.apply(image)
    return image

def rotate_image_around_point(image, xy, angle):
    """
    Rotate an image around a point by an angle in counter-clockwise direction.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    im_rot = rotate(image,angle) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    
    return im_rot, new+rot_center

def homography_transform(alignment_points_ref: np.ndarray, 
                         alignment_points_warp: np.ndarray):
    """
    Compute the homography transformation matrix from the alignment points.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    # Convert the list of points to numpy array
    pass
