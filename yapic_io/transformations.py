'''
matrix transformation functions
'''

import numpy as np 
from skimage import transform as tf
import logging
from scipy.ndimage import map_coordinates
logger = logging.getLogger('utils')
        
def mirror_edges(mat, n_pix):
    '''
    increase matrix at all edges by mirroring
    by n_pix pixels. n_pix is a vector with one value for each dimension

    :param mat: n-dimensional matrix
    :type mat: numpy.ndarray
    :param n_pix: nr of edge pixels to be mirrored for each dimension 
    :type n_pix: tuple of integers
    :returns:  numpy.ndarray -- the matrix with mirrored edges.
    '''

    n_pix = [(e, e) for e in n_pix]
    
    return np.pad(mat, n_pix, mode='reflect')    


def get_transform(image, rotation_angle, shear_angle):
    '''
    Returns:
      Transformation object for centered rotation and shear of a 2D matrix

    :param  image: 2-dimensional matrix
    :type   image: numpy.ndarray
    :param  rotation_angle: angle in degrees
    :type   rotation_angle: float
    :param  shear_angle: angle in degrees
    :type   shear_angle: float
    :returns:  transformation object .
    '''
    if len(image.shape) != 2:
        raise ValueError('image has %s dimensions instead of 2' \
                            % len(image.shape))


    shift_y, shift_x = np.array(image.shape[:2]) / 2.
    tf_rotate_shear = tf.AffineTransform(rotation=np.deg2rad(rotation_angle)\
        , shear=np.deg2rad(shear_angle))
    tf_shift = tf.AffineTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.AffineTransform(translation=[shift_x, shift_y])
    tf_center_rot = (tf_shift + (tf_rotate_shear + tf_shift_inv)).inverse

    return tf_center_rot    


def warp_image_2d(image, rotation_angle, shear_angle):
    '''
    Warps 2d matrix with affine transform (rotation and shear
    relative to image center).
    Empty edges are filled by mirroring.

    :param image: 2-dimensional matrix
    :type  image: numpy.ndarray
    :param rotation_angle: angle in degrees
    :type  rotation_angle: float
    :param shear_angle: angle in degrees
    :type  shear_angle: float
    :returns:  transformed 2d matrix

    '''
    if len(image.shape) != 2:
        raise ValueError('image has %s dimensions instead of 2'\
                                % len(image.shape))

    t = get_transform(image, rotation_angle, shear_angle)
    coords = tf.warp_coords(t, image.shape) 
    
    return map_coordinates(image, coords, order=0, mode='reflect')    









