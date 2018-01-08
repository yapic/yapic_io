import os
import numpy as np
import logging
import os
import random
import itertools
logger = logging.getLogger(os.path.basename(__file__))
from difflib import SequenceMatcher
from munkres import Munkres


def assign_slice_by_slice(assignment_dicts, vol):
    # potential bug? we must assign to a copy of vol!
    for c, cd in enumerate(assignment_dicts):
        for key, value in cd.items():
            vol[c][vol[c] == key] = value
    return vol


def get_tile_meshgrid(image_shape, pos, size):
    '''
    returns coordinates for selection of a sub matrix,
    given a certain n-dimensional position (upper left)
    and n-dimensional size.

    :param shape: tuple defining the image shape
    :param pos: tuple defining the upper left position of the tile in n dimensions
    :param size: tuple defining size of tile in all dimensions
    :returns: a multidimensional meshgrid defining the sub matrix

    '''
    assert_valid_image_subset(image_shape, pos, size)
    return [slice(p, p+s) for p, s in zip(pos, size)]

def assert_valid_image_subset(image_shape, pos, size):
    '''
    check if a requested image subset defined by upper left pos and size
    fits into the image of given shape
    '''
    pos = np.array(pos)

    np.testing.assert_equal(len(image_shape), len(size))
    np.testing.assert_equal(len(image_shape), len(pos))
    assert not (pos < 0).any(), 'tile out of image bounds'
    assert not (image_shape < (pos + size)).any(), 'tile out of image bounds'

def compute_pos(img_shape, tile_shape):
    '''
    computes all possible positions for fetching tiles of a given size
    if the tiles do not fit perfectly in the image, the last positions
    are corrected, such that the last and the second-last tile would
    have some overlap.
    '''
    img_shape = np.array(img_shape)
    tile_shape = np.array(tile_shape)

    assert not (tile_shape > img_shape).any(), 'tile size {} > image shape {}'.format(tile_shape, img_shape)

    pos_per_dim = [ np.arange(0, length, step)
                    for length, step in zip(img_shape, tile_shape) ]

    mod = (img_shape % tile_shape) # nr of out of bounds pixels for last tile
    shift_last_tile = mod - tile_shape
    shift_last_tile[mod == 0] = 0

    for el, shift in zip(pos_per_dim, shift_last_tile):
        el[-1] = el[-1] + shift

    pos = list(itertools.product(*pos_per_dim))
    return pos


def flatten(listOfLists):
    "Flatten one level of nesting"
    return itertools.chain.from_iterable(listOfLists)


def add_to_filename(path, insert_str):
    '''
    adds suffix ore prefix to filename

    >>> from yapic_io.utils import add_to_filename
    >>> path = 'path/to/tiff/file.tif'
    >>> add_str = 'label_1'
    >>> add_to_filename(path, add_str)
    'path/to/tiff/file_label_1.tif'
    '''
    assert insert_str.replace('_', '').isalnum(), '"{}" is not alphanumeric'.format(insert_str)

    path, ext = os.path.splitext(path)
    return '{}_{}{}'.format(path, insert_str, ext)


def compute_str_dist_matrix(s1, s2):
    '''
    - compute matrix of string distances for two lists of strings
    - normalize lengths of string lists (fill shorter list with empty strings)
    '''
    s1 = s1 + ['' for _ in range(len(s2) - len(s1))]
    s2 = s2 + ['' for _ in range(len(s1) - len(s2))]

    mat = np.zeros((len(s1), len(s2)))
    for i, a in enumerate(s1):
        for j, b in enumerate(s2):
            mat[i, j] = 1 - SequenceMatcher(None, a, b).ratio()

    return mat, s1, s2


def find_best_matching_pairs(s1, s2):
    '''
    find global minimum for pairwise assignment of strings
    by using the munkres (hungarian) algorithm
    '''
    if len(s1) == 0:
        assert len(s2) == 0
        return []

    mat, s1norm, s2norm = compute_str_dist_matrix(s1, s2)

    m = Munkres()
    indexes = m.compute(mat) # find assignment combination with lowest global cost

    pairs = [ [s1norm[i], s2norm[j]] for i, j in indexes]

    # change empty strings to None
    for pair in pairs:
        if pair[0] == '':
            pair[0] = None
        elif pair[1] == '':
            pair[1] = None

    return [pair for pair in pairs if pair[0] is not None]
