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
    return tf.warp(image, t, order=0, mode='symmetric', preserve_range=True) # 0.765s

def warp_image_2d_stack(image, rotation_angle, shear_angle):
    '''
    Warps a 3d or 4d matrix (stack by stack) and slice by slice with affine transform (rotation and shear
    relative to image center) in 2D.
    Empty edges are filled by mirroring.

    :param image: 3-dimensional matrix or 4-dimensional matrix
    :type  image: numpy.ndarray
    :param rotation_angle: angle in degrees
    :type  rotation_angle: float
    :param shear_angle: angle in degrees
    :type  shear_angle: float
    :returns:  transformed 3d matrix

    '''


    if (len(image.shape) != 3) and (len(image.shape) != 4):
        raise ValueError('image has %s dimensions instead of 3 or 4'\
                                % len(image.shape))

    if (len(image.shape) == 3):

        return np.array([warp_image_2d(z_slice, rotation_angle, shear_angle) for z_slice in image])

    if (len(image.shape) == 4):
        out = []

        for image_zxy in image:
            out.append([warp_image_2d(z_slice, rotation_angle, shear_angle) for z_slice in image_zxy])
        return np.array(out)


def flip_image_2d_stack(image, fliplr=False, flipud=False, rot90=0):
    '''
    Flips and rotates a zxy stack in xy with fast numpy operations
    '''
    image = image.T
    if fliplr:
        image = np.fliplr(image)
    if flipud:
        image = np.flipud(image)
    if rot90>0:
        image = np.rot90(image, k=rot90)
    return image.T


def calc_warping_shift(image_shape, rotation_angle, shear_angle):

    z = np.zeros(image_shape)
    mat = np.arange(image_shape[0]*image_shape[1]).reshape((image_shape))
    center_pos = (np.array(image_shape)-1)/2

    z[:, center_pos[1]] = 111
    z[center_pos[0], :] = 256
    z[center_pos[0], center_pos[1]-2:center_pos[1]+2] = 256
    m2 = warp_image_2d(mat, rotation_angle, shear_angle)
    z_rot = warp_image_2d(z, rotation_angle, shear_angle)
    #print(mat)
    #print(m2)
    #print(z)
    #print(z_rot)





