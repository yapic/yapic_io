import numpy as np
import logging
import os
import itertools
from difflib import SequenceMatcher
from munkres import Munkres
import sys
logger = logging.getLogger(os.path.basename(__file__))


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)

    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x),
                   j,
                   count))
        file.flush()
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()


def get_tile_meshgrid(image_shape, pos, size):
    '''
    Returns coordinates for selection of a sub matrix, given a certain
    n-dimensional position (upper left) and n-dimensional size.

    Parameters
    ----------
    image_shape : tuple of size ndim
        Image shape.
    pos: tuple
        Upper left position of the tile in n dimensions.
    size : tuple of length ndim
        Size of tile.

    Returns
    -------
    array_like
        Multidimensional meshgrid defining the sub matrix
    '''
    assert_valid_image_subset(image_shape, pos, size)
    return [slice(p, p+s) for p, s in zip(pos, size)]


def assert_valid_image_subset(image_shape, pos, size):
    '''
    Check if a requested image subset defined by upper left pos and size
    fits into the image of given shape.

    Parameters
    ----------
    image_shape : tuple of size ndim
        Image shape.
    pos: tuple
        Upper left position of the tile in n dimensions.
    size : tuple of length ndim
        Size of tile.

    Raises
    ------
        AssertionError
    '''
    pos = np.array(pos)

    np.testing.assert_equal(len(image_shape), len(size))
    np.testing.assert_equal(len(image_shape), len(pos))
    assert not (pos < 0).any(), 'tile out of image bounds'
    assert not (image_shape < (pos + size)).any(), 'tile out of image bounds'


def compute_pos(img_shape, tile_shape, sliding=None):

    '''
    Computes all  positions for fetching tiles of a given size (tile
    window).

    If the tiles do not fit perfectly in the image, the last positions
    are corrected, such that the last and the second-last tile would
    have some overlap.

    Parameters
    ----------
    img_shape : tuple of size ndim
        Image shape.
    tile_shape : tuple of size ndim
        Tile shape.
    sliding : None or tuple
        if tuple, positions are calculated for overlapping tiles. The tile
        positions are shifted by  n pixels in each direction as defined in the
        tuple.
        If None, positions are calculated for non overlapping tiles. However
        the last tile could overlap with its neighbors.
    shifted : integer
        Nr of pixels to shift the sliding window in all dimensions.

    Returns
    -------
    array_like
        list of n-dimensional positions
    '''
    img_shape = np.asarray(img_shape)
    tile_shape = np.asarray(tile_shape)

    assert sliding is None or len(sliding) == len(img_shape)

    msg = 'tile size {} > image shape {}'.format(tile_shape, img_shape)
    assert not (tile_shape > img_shape).any(), msg

    pos_per_dim = [np.arange(0, length, step)
                   for length, step in zip(img_shape, tile_shape)]

    mod = (img_shape % tile_shape)  # nr of out of bounds pixels for last tile
    shift_last_tile = mod - tile_shape
    shift_last_tile[mod == 0] = 0

    for el, shift in zip(pos_per_dim, shift_last_tile):
        el[-1] = el[-1] + shift

    pos = list(itertools.product(*pos_per_dim))

    if sliding is None or len(pos) <= 1:
        # non overlapping tile positions
        return np.array(pos)

    # if sliding window: interpolate missing positions with mgrid
    n_dims = len(img_shape)

    slices = [slice(pos[0][i], pos[-1][i]+1, s)
              for s, i in zip(sliding, range(n_dims))]
    mesh = np.mgrid[slices]
    mesh = np.array(mesh).swapaxes(0, -1)

    n_dims = len(img_shape)
    pos_array = mesh.reshape((np.int(mesh.size/n_dims), n_dims))

    return pos_array


def find_overlapping_tiles(a, pos, shape):

    a = np.asarray(a)
    pos = np.asarray(pos)

    is_overlap = []
    for dim in range(pos.shape[-1]):
        is_overlap.append(
            np.stack([pos[:, dim] <= a[dim] + shape[dim]-1,
                      pos[:, dim] + shape[dim]-1 >= a[dim]]).all(axis=0))
    return np.stack(is_overlap).all(axis=0)


def segregate_tile_pos(pos, shape, choices):
    '''
    splits a vector of positions in two vectors and removes all overlapping
    posisions.
    '''
    pos = np.asarray(pos)

    p1 = np.delete(pos, choices, axis=0)
    p2 = pos[choices, :]

    for a in p2:
        is_overlap_ind = find_overlapping_tiles(a, p1, shape)
        p1 = np.delete(p1, np.nonzero(is_overlap_ind)[0], axis=0)

    return [tuple(e) for e in p1], [tuple(e) for e in p2]


def _compute_str_dist_matrix(s1, s2):
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
    Find global minimum for pairwise assignment of strings
    by using the munkres (hungarian) algorithmself.

    Parameters
    ----------
    s1 : array_like
        List of strings.
    s2 : array_like
       List of strings.

    Notes
    -----
    Counts of non-empty elements must be equal for s1 and s2.
    '''

    # remove empties
    s1 = list(filter(None, s1))
    s2 = list(filter(None, s2))

    if len(s1) == 0:
        assert len(s2) == 0
        return []

    mat, s1norm, s2norm = _compute_str_dist_matrix(s1, s2)

    # find assignment combination with lowest global cost
    m = Munkres()
    indexes = m.compute(mat)

    pairs = [[s1norm[i], s2norm[j]] for i, j in indexes]

    # change empty strings to None
    for pair in pairs:
        if pair[0] == '':
            pair[0] = None
        elif pair[1] == '':
            pair[1] = None

    return [pair for pair in pairs if pair[0] is not None]
