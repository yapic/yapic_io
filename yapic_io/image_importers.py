#from skimage.io import imread
from PIL import Image, ImageSequence
import logging
import os
import numpy as np
logger = logging.getLogger(os.path.basename(__file__))


def get_tiff_image_dimensions(path):
    '''
    returns dimensions of a single image file:

    :param  path: path to image file
    :type path: str
    :returns tuple with integers (nr_channels, nr_zslices, nr_x, nr_y)
    '''
    
    im = Image.open(path)

    z = im.n_frames
    x = im.width
    y = im.height

    if im.mode == '1' \
        or im.mode == 'L' \
        or im.mode == 'P' \
        or im.mode == 'I' \
        or im.mode == 'F':
        c = 1

    if im.mode == 'RGB':
        c = 3
    if im.mode == 'RGBA':
        c = 4    
    if im.mode == 'CMYK':
        c = 4    
    if im.mode == 'YCbCr':
        c = 3    

    return (c, z, x, y) 


def import_tiff_image(path):
    '''
    imports image file as numpy array
    in following dimension order:
    (channel, zslice, x, y)

    :param  path: path to image file
    :type path: str
    :returns 4 dimensional numpy array (channel, zslice, x, y)
    '''
    
    dims = get_tiff_image_dimensions(path)
    
    im = Image.open(path)

    image = np.array([np.array(frame) for frame in ImageSequence.Iterator(im)])
    
    # add dimension for channel
    if dims[0] == 1:
        image = np.expand_dims(image,axis=-1)

    # bring dimensions in right order
    # (z,y,x,c) -> (c,z,x,y)
    image = np.swapaxes(image,3,0) # (z,y,x,c) -> (c,y,x,z)
    image = np.swapaxes(image,1,3) # (c,y,x,z) -> (c,z,x,y)
    
    return image

    
    


        

