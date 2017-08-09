#from skimage.io import imread
from PIL import Image, ImageSequence
import logging
import os
import numpy as np
import yapic_io.utils as ut
from tifffile import imsave, imread, TiffFile
from functools import lru_cache


logger = logging.getLogger(os.path.basename(__file__))

@lru_cache(maxsize = 5000)
def is_rgb_image(path):
    with TiffFile(path) as tiff:
        tag = next(page.tags['photometric'] for page in tiff)
        return tag.value == 2


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
    with TiffFile(path) as tiff:
        series = tiff.series[0]
        series_shape = np.array(series.shape)

        shape = [ series_shape[src] if src is not None else 1
                  for src in axes_order(series.axes)]

    return tuple(shape)


def index(array, value):
    '''
    Like `array.index(value)` but returns None if value not found.
    (`array.index(value)` would otherwise raise a ValueError)
    '''
    try:
        return array.index(value)
    except ValueError:
        return None


def axes_order(input_axes):
    '''
    Determines how to reorder the axes to get the standardized 'CZXY' format.
    Returns a 4-tuple with None if the dimension does not exist in the input_axes
    '''
    # tifffile.AXES_LABELS = {
    #     'X': 'width',
    #     'Y': 'height',
    #     'Z': 'depth',
    #     'S': 'sample',  # rgb(a)
    #     'I': 'series',  # general sequence, plane, page, IFD
    #     'T': 'time',
    #     'C': 'channel',  # color, emission wavelength
    #     'A': 'angle',
    #     'P': 'phase',  # formerly F    # P is Position in LSM!
    #     'R': 'tile',  # region, point, mosaic
    #     'H': 'lifetime',  # histogram
    #     'E': 'lambda',  # excitation wavelength
    #     'L': 'exposure',  # lux
    #     'V': 'event',
    #     'Q': 'other',
    #     'M': 'mosaic',  # LSM 6
    # }

    C = index(input_axes, 'C')
    if C is None:
        C = index(input_axes, 'S')

    Z = index(input_axes, 'Z')
    if Z is None:
        Z = index(input_axes, 'I')

    Y = index(input_axes, 'Y')
    X = index(input_axes, 'X')

    assert Y is not None
    assert X is not None

    return (C,Z,X,Y)


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
    with TiffFile(path) as tiff:
        series = tiff.series[0]
        img = series.asarray()

        new_order = np.array(axes_order(series.axes))

        img = np.transpose(img, [d for d in new_order if d is not None])

        for i, dim in enumerate(new_order):
            if dim is None:
                img = np.expand_dims(img, i)

    assert len(img.shape) == 4
    return img


def init_empty_tiff_image(path, x_size, y_size, z_size=1):
    '''
    initializes dark 32 bit floating point grayscale tiff image
    '''

    path = autocomplete_filename_extension(path)
    check_filename_extension(path)

    data = np.zeros((z_size, x_size, y_size), dtype=np.float32)
    logger.info('write empty tiff image with %s values to %s', data.dtype, path)

    assert len(data.shape) == 3
    imsave(path, data, metadata={'axes': 'ZXY'})


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
    assert len(img.shape) == 4

    pos_czxy = (0, ) + pos_zxy
    size_czxy = (1, ) + pixels.shape
    mesh = ut.get_tile_meshgrid(img.shape, pos_czxy, size_czxy)

    img[mesh] = pixels

    assert len(img.shape) == 4
    imsave(path, img[0,...], metadata={'axes': 'ZXY'})


def check_filename_extension(path, format_str='.tif'):
    ext = os.path.splitext(path)[1]

    if ext != format_str:
        raise ValueError('extension must be %s, but is %s', (format_str, ext))


def autocomplete_filename_extension(path, format_str='.tif'):
    ext = os.path.splitext(path)[1]
    if ext == '':
        return path + format_str
    return path
