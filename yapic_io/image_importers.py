#from skimage.io import imread
from PIL import Image, ImageSequence
import logging
import os
import numpy as np
import yapic_io.utils as ut
from tifffile import imsave, imread
from functools import lru_cache


logger = logging.getLogger(os.path.basename(__file__))

@lru_cache(maxsize = 5000)
def is_rgb_image(path):
    try:
        im = Image.open(path)
    except OSError:
        return True
    if im.mode == 'RGB':
        return True
    return False

@lru_cache(maxsize = 5000)
def get_tiff_image_dimensions(path, multichannel=None, zstack=None):
    '''
    returns dimensions of a single image file:

    if nr_channels is None, the importer tries to find out the nr of channels
    (this works e.g. with rgb images)

    :param  path: path to image file
    :type path:
    :param nr_channels: nr of channels (to allow correct mapping for z and channels)
    :returns tuple with integers (nr_channels, nr_zslices, nr_x, nr_y)
    '''

    #return import_tiff_image_2(path, third_dim_as_z=third_dim_as_z).shape
    # im = Image.open(path)

    # z = im.n_frames
    # x = im.width
    # y = im.height
    # print(im.mode)
    # if im.mode == '1' \
    #     or im.mode == 'L' \
    #     or im.mode == 'P' \
    #     or im.mode == 'I' \
    #     or im.mode == 'F':
    #     c = 1
    #     if nr_channels:
    #         c = nr_channels
    #         z = z/nr_channels

    # if im.mode == 'RGB':
    #     c = 3
    # if im.mode == 'RGBA':
    #     c = 4
    # if im.mode == 'CMYK':
    #     c = 4
    # if im.mode == 'YCbCr':
    #     c = 3

    img = import_tiff_image(path, multichannel=multichannel, zstack=zstack)

    if img is None:
        raise ValueError('Image dimensions could not be detected')

    return img.shape



def import_tiff_image(path, multichannel=None, zstack=None):
    '''
    Returns a tiff image as numpy array with dimensions (c, z, x, y).

    :param path: path to file
    :param multichannel: if True, the image is interpreted as multichannel image in unclear cases
    :param zstack: if True, the image is interpreted as multichannel image in unclear cases

    multichannel and zstack can be set to None. In this case the importer tries to
    map dimensions automatically. This does not always work, esp. in case of 3
    dimensional images.


    Examples:

    - If zstack is set to False and multichannel is set to None,
      the importer will assign the thrid dimensions (in case of 3 dimensional images)
      to channels, i.e. interprets the image as multichannel, single z image.

    - If zstack is set to None and multichannel is set to None,
      the importer will assign all dims correctly in case of 4 dimensional images
      and in case of 2 dimensional images (single z, singechannel). In case of 3
      dimensional images, it throws an error, because it is not clear if the thrid
      dimension is z or channel (RGB images will still be mapped correctly)

    '''

    img = imread(path)

    n_dims = len(img.shape)
    if n_dims>4:
        return IOError('can not read images with more than 4 dimensions (czxy)')

    if n_dims == 2: #case one z, one channel
        img = np.swapaxes(img, 0, 1) # (y, x) -> (x, y)
        img = np.expand_dims(img, axis=0) # (x, y) -> (z, x, y)
        img = np.expand_dims(img, axis=0) # (z, x, y) -> (c, z, x, y)
        return img

    rgb = is_rgb_image(path)

    if n_dims == 3 and rgb: #case one z, three channels in rgb mode
        img = np.swapaxes(img, 0, 2) # (y, x, c) -> (c, x, y)
        img = np.expand_dims(img, axis=1) # (c, x, y) -> (c, z, x, y)
        return img

    #try to determine multichannel or multi_z automatically in case of 3 dims
    if n_dims == 3 and not rgb and multichannel is None and zstack==False: #case one z, >1 channels in grayscale mode
        multichannel = True
    if n_dims == 3 and not rgb and multichannel==False and zstack is None: #case >1z, one channelsin grayscale mode
        zstack = True

    if n_dims == 3 and not rgb and multichannel and zstack:
        raise ValueError('Either multichannel or zstack has to be set to False')

    if n_dims == 3 and not rgb and not multichannel and not zstack:
        raise ValueError('Either multichannel or zstack has to be set to True')



    if n_dims == 3 and not rgb and multichannel and not zstack: #case one z, >1 channels in grayscale mode
        img = np.swapaxes(img, 1, 2) # (c, y, x) -> (c, x, y)
        img = np.expand_dims(img, axis=1) # (c, x, y) -> (c, z, x, y)
        return img

    if n_dims == 3 and not rgb and zstack and not multichannel: #case >1 z, 1 channel in grayscale mode
        img = np.swapaxes(img, 1, 2) # (z, y, x) -> (z, x, y)
        img = np.expand_dims(img, axis=0) # (z, x, y) -> (c, z, x, y)
        return img


    if n_dims == 4 and rgb: #case >1 z, three channel in rgb mode
        img = np.swapaxes(img, 1, 3) # (z, y, x, c) -> (z, c, x, y)
        img = np.swapaxes(img, 0, 1) # (z, c, x, y) -> (c, z, x, y)
        return img

    if n_dims == 4 and not rgb: #case >1 z, >1 channel in grayscale mode
        img = np.swapaxes(img, 2, 3) # (z, c, y, x) -> (z, c, x, y)
        img = np.swapaxes(img, 0, 1) # (z, c, x, y) -> (c, z, x, y)
        return img



# def import_tiff_image(path, nr_channels=None):
#     '''
#     imports image file as numpy array
#     in following dimension order:
#     (channel, zslice, x, y)



#     :param  path: path to image file
#     :type path: str
#     :returns 4 dimensional numpy array (channel, zslice, x, y)
#     '''

#     dims = get_tiff_image_dimensions(path)

#     im = Image.open(path)

#     image = np.array([np.array(frame) for frame in ImageSequence.Iterator(im)])
#     # image = imread(path)
#     # print(path)
#     # print('shape')
#     # print(image.shape)
#     # print('dims')
#     # print(dims)
#     # add dimension for channel
#     if dims[0] == 1:
#         image = np.expand_dims(image, axis=-1)

#     # bring dimensions in right order
#     # (z, y, x, c) -> (c, z, x, y)
#     image = np.swapaxes(image, 3, 0) # (z, y, x, c) -> (c, y, x, z)
#     image = np.swapaxes(image, 1, 3) # (c, y, x, z) -> (c, z, x, y)

#     return image


def init_empty_tiff_image(path, x_size, y_size, z_size=1):
    '''
    initializes dark 32 bit floating point grayscale tiff image
    '''

    path = autocomplete_filename_extension(path)
    check_filename_extension(path)



    data = np.zeros((z_size, y_size, x_size), dtype=np.float32)
    logger.info('write empty tiff image with %s values to %s'\
        , type(data[0, 0, 0]), path)
    #print(type(data[0, 0]))
    #img = Image.fromarray(data)
    #img.save(path)
    imsave(path, data, imagej=True)


def add_vals_to_tiff_image(path, pos_zxy, pixels):
    '''
    opens a onechannel zstack tiff image, overwrites pixels values
    with pixels at pos_zxy
    and overwrites the input tiff image with the new pixels.
    pixels and pos_zxy are in order (z, x, y)

    '''

    pixels = np.array(pixels, dtype=np.float32)

    path = autocomplete_filename_extension(path)
    check_filename_extension(path)

    img = import_tiff_image(path, zstack=True)

    pos_czxy = (0, ) + pos_zxy
    size_czxy = (1, ) + pixels.shape
    mesh = ut.get_tile_meshgrid(img.shape, pos_czxy, size_czxy)

    img[mesh] = pixels
    img = np.squeeze(img, axis=0) #remove channel axis, which should be 1
    img = np.swapaxes(img, 2, 1) # (z, x, y) -> (z, y, x)

    #imsave(path, img)
    #im = Image.fromarray(img)

    #im.save(path)
    try:
        logger.debug('updating image %s...', path)
        imsave(path, img, imagej=True)
    except:
        return False




def check_filename_extension(path, format_str='.tif'):
    ext = os.path.splitext(path)[1]


    if ext != format_str:
        raise ValueError('extension must be %s, but is %s', (format_str, ext))

def autocomplete_filename_extension(path, format_str='.tif'):
    ext = os.path.splitext(path)[1]
    if ext == '':
        return path + format_str
    return path
