'''
matrix transformation functions
'''

import numpy as np
from skimage import transform as tf
import logging

logger = logging.getLogger('utils')


def get_transform(image, rotation_angle, shear_angle):
    '''
    Get transformation object for centered rotation and shear of a 2D matrix.

    Paramters
    ---------
    image : numpy.ndarray
        2-dimensional matrix.
    rotation_angle : float
        Angle in degrees.
    shear_angle: float
        angle in degrees

    Returns
    -------
    transformation object
    '''
    if len(image.shape) != 2:
        msg = 'Image has {} dimensions, must have 2.'.format(image.ndim)
        raise ValueError(msg)

    shift_y, shift_x = np.array(image.shape[:2]) / 2.
    tf_rotate_shear = tf.AffineTransform(rotation=np.deg2rad(rotation_angle),
                                         shear=np.deg2rad(shear_angle))
    tf_shift = tf.AffineTransform(translation=[-shift_x, -shift_y])
    tf_shift_inv = tf.AffineTransform(translation=[shift_x, shift_y])
    tf_center_rot = (tf_shift + (tf_rotate_shear + tf_shift_inv)).inverse

    return tf_center_rot


def warp_image_2d(image, rotation_angle, shear_angle):
    '''
    Warps 2d matrix with affine transform.

    Rotation and shear axis is in image center.
    Empty edges are filled by mirroring.

    Paramters
    ---------
    image : numpy.ndarray
        2-dimensional matrix
    rotation_angle: float
        angle in degrees
    shear_angle: float
        angle in degrees

    Returns
    -------
    numpy.ndarray
        transformed 2d matrix
    '''
    if len(image.shape) != 2:
        msg = 'Image has {} dimensions, must have 2.'.format(image.ndim)
        raise ValueError(msg)

    t = get_transform(image, rotation_angle, shear_angle)
    return tf.warp(image, t, order=0, mode='symmetric', preserve_range=True)

def warp_image_2d_stack(image, rotation_angle, shear_angle):
    '''
    Warps a 3d or 4d matrix with affine transform.

    Transformation is applied in 2D slice by sclice.
    Rotation and shear axis is in image center.
    Empty edges are filled by mirroring.

    Paramters
    ---------
    image : numpy.ndarray
        3-dimensional matrix or 4-dimensional matrix
    rotation_angle : float
        angle in degrees
    shear_angle : float
        angle in degrees

    Returns
    -------
    nump.ndarray
        transformed 3d matrix
    '''

    if image.ndim == 3:
        return np.array([warp_image_2d(z_slice, rotation_angle, shear_angle)
                         for z_slice in image])

    elif image.ndim == 4:
        out = []
        for image_zxy in image:
            out.append([warp_image_2d(z_slice, rotation_angle, shear_angle)
                        for z_slice in image_zxy])
        return np.array(out)
    msg = 'Image has {} dimensions, must have 3 or 4.'.format(image.ndim)
    raise ValueError(msg)


def flip_image_2d_stack(image, fliplr=False, flipud=False, rot90=0):
    '''
    Flips and rotates a zxy stack in xy with fast numpy operations.

    Parameters
    ----------
    image : array_like
        must be at least 2D
    fliplr : bool, optional
        If True, image is flipped vertically.
    fliplr : bool, optional
        If True, image is flipped upside down.
    rot90 : integer
        Number of times the array is rotated by 90 degrees.

    Returns
    -------
    numpy.ndarray
        transformed image
    '''
    image = image.T
    if fliplr:
        image = np.fliplr(image)
    if flipud:
        image = np.flipud(image)
    if rot90 > 0:
        image = np.rot90(image, k=rot90)
    return image.T
