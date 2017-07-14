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
    nr_channels = vol.shape[0]
    for c, cd in zip(range(nr_channels), assignment_dicts):
        for key in cd.keys():
            vol[c][vol[c]==key] = cd[key]
    return vol




def remove_exclusive_vals_from_set(list_of_sets):
    '''
    takes a list of sets and removes for each set the values
    that are not present any of the other sets of the list.

    example:
    list_of_sets = [{1, 2, 3}, {1, 2}, {1, 4}]
    remove_exclusive_vals_from_set(list_of_sets)
    [{1, 2}, {1, 2}, {1}]

    '''

    out = []
    for i in range(len(list_of_sets)):
        subset = set()
        for j in range(len(list_of_sets)):
            if j==i:
                pass
            else:
                subset = subset.union(list_of_sets[j])
        out.append(list_of_sets[i].intersection(subset))
    return out

def get_exclusive_vals_from_set(list_of_sets):
    '''
    takes a list of sets and removes for each set the values
    that are also present in the other sets of the list.

    example:
    list_of_sets = [{1, 2, 3}, {1, 2}, {1, 4}]
    remove_exclusive_vals_from_set(list_of_sets)
    [{3}, set(), {4}]

    '''

    out = []
    for i in range(len(list_of_sets)):
        subset = set()
        for j in range(len(list_of_sets)):
            if j==i:
                pass
            else:
                subset = subset.union(list_of_sets[j])
        out.append(list_of_sets[i].difference(subset))
    return out



def get_indices(pos, size):
    '''
    returns all indices for a sub matrix, given a certain n-dimensional position (upper left)
    and n-dimensional size.


    :param pos: tuple defining the upper left position of the tile in n dimensions
    :param size: tuple defining size of tile in all dimensions
    :returns: list of indices for all dimensions
    '''
    if len(pos) != len(size):
        error_str = '''nr of dimensions does not fit: pos(%s),
            size (%s)''' % (pos, size)
        raise ValueError(error_str)


    return [list(range(p, p+s)) for p, s in zip(pos, size)]


def get_indices_fast(pos, size):
    '''
    returns all indices for a sub matrix, given a certain n-dimensional position (upper left)
    and n-dimensional size.


    :param pos: tuple defining the upper left position of the tile in n dimensions
    :param size: tuple defining size of tile in all dimensions
    :returns: list of indices for all dimensions
    '''
    # if len(pos) != len(size):
    #     error_str = '''nr of dimensions does not fit: pos(%s),
    #         size (%s)''' % (pos, size)
    #     raise ValueError(error_str)



    return [list(np.arange(p, p+s)) for p, s in zip(pos, size)]





def get_tile_meshgrid(image_shape, pos, size):
    '''
    returns coordinates for selection of a sub matrix
    , given a certain n-dimensional position (upper left)
    and n-dimensional size.

    :param shape: tuple defining the image shape
    :param pos: tuple defining the upper left position of the tile in n dimensions
    :param size: tuple defining size of tile in all dimensions
    :returns: a multidimensional meshgrid defining the sub matrix

    '''

    pos = np.array(pos)
    size = np.array(size)

    if not is_valid_image_subset(image_shape, pos, size):
        raise ValueError('image subset not valid')

    return [slice(p, p+s) for p, s in zip(pos, size)]

    indices = get_indices_fast(pos, size)
    return np.meshgrid(*indices, indexing='ij')


def is_valid_image_subset(image_shape, pos, size):
    '''
    check if a requested image subset defined by upper left pos and size
    fits into the image of giveb sahpe
    '''

    pos = np.array(pos)
    size = np.array(size)

    if len(image_shape) != len(size):
        error_str = '''nr of image dimensions (%s)
            and size vector length (%s) do not match'''\
            % (len(image_shape), len(size))
        logger.error(error_str)
        return False
        # raise ValueError(error_str)

    if len(image_shape) != len(pos):
        error_str = '''nr of image dimensions (%s)
            and pos vector length (%s) do not match'''\
            % (len(image_shape), len(pos))
        logger.error(error_str)
        return False
        # raise ValueError(error_str)


    if (image_shape < (pos + size)).any():
        error_str = '''tile out of image bounds: image shape: %s,
            pos: %s, tile size: %s''' % (image_shape, pos, size)
        logger.error(error_str)
        return False
        # raise ValueError(error_str)

    if (pos < 0).any():
        error_str = '''tile out of image bounds: image shape: %s,
            pos: %s, tile size: %s''' % (image_shape, pos, size)
        logger.error(error_str)
        return False
    return True


def flatten_label_coordinates(label_coordinates):
    return [(label, coor) for label in label_coordinates.keys() for coor in label_coordinates[label]]


def get_max_pos_for_tile(size, shape):
    '''
    returns maxpos as np array
    '''
    
    return np.array(shape) - np.array(size)


def get_random_pos_for_coordinate(coor, size, shape):

    if len(coor) != len(size):
        raise ValueError('coor and size must have same lengths. coor: %s, size: %s' \
            % (str(coor), str(size)))

    coor = np.array(coor)
    size = np.array(size)


    maxpos = get_max_pos_for_tile(size, shape)
    maxpos[maxpos>coor] = coor[maxpos>coor]

    minpos = coor - size + 1
    minpos[minpos < 0] = 0

    random_pos = []
    for maxdim, mindim in zip(maxpos, minpos):
        dim_range = range(mindim, maxdim+1)
        random_pos.append(random.choice(dim_range))

    return tuple(random_pos)


def compute_pos(shape, size):
    '''
    computes all possible positions for fetching tiles of a given size
    if the tiles do not fit perfectly in the image, the last positions
    are corrected, such that the last and the second-last tile would
    have some overlap.
    '''
    shape = np.array(shape)
    size = np.array(size)

    if (size > shape).any():
        raise ValueError('tile size is larger than image shape. size: %s, shape: %s' \
            % (str(size), str(shape)))

    nr_tile = np.ceil(shape/size)
    shift_last_tile = np.zeros(len(shape)).astype(int)
    mod = shape % size # nr of out of bounds pixels for last tile

    if mod.any():
        shift_last_tile[mod!=0] = mod[mod!=0] - size[mod!=0]

    pos_per_dim = \
        [list(range(0, imlength, templength)) for imlength, templength in zip(shape, size)]

    for el, shift in zip(pos_per_dim, shift_last_tile):
        el[-1] = el[-1] + shift

    pos = list(itertools.product(*pos_per_dim))
    return pos


def add_to_filename(path, insert_str, suffix=True):
    '''
    adds suffix ore prefix to filename

    >>> from yapic_io.utils import add_to_filename
    >>> path = 'path/to/tiff/file.tif'
    >>> add_str = 'label_1'
    >>> add_to_filename(path, add_str, suffix=True)
    'path/to/tiff/file_label_1.tif'
    >>> add_to_filename(path, add_str, suffix=False) # add as prefix
    'path/to/tiff/label_1_file.tif'

    '''

    if not insert_str.replace('_', '').isalnum():
        # underscore and alphanumeric characters are allowed
        raise ValueError('insert_str characters must only be alphanumeric, not the case here: %s' \
            % insert_str)


    filename = os.path.basename(path)
    dirname = os.path.dirname(path)
    filename_trunk, ext = os.path.splitext(filename)

    if suffix:
        out_filename = filename_trunk + '_' + insert_str + ext
    else:
        out_filename = insert_str + '_' + filename_trunk + ext
    return os.path.join(dirname, out_filename)





def string_distance(a, b):
    return 1-SequenceMatcher(None, a, b).ratio()

def compute_str_dist_matrix(s1, s2):
    '''
    -compute matrix of string distances for two lists of strings
    - normalize lengths of string lists (fill shorter list with empty strings)
    '''
    lendiff = len(s1)-len(s2)

    appendix = ['' for _ in range(abs(lendiff))]
    if lendiff < 0: # if s2 is longer:
        s1 = s1 + appendix
    if lendiff >0: # if s1 is longer:
        s2 = s2 + appendix


    mat = np.zeros((len(s1), len(s2)))

    for i in range(len(s1)):
        for j in range(len(s2)):
            mat[i, j] = string_distance(s1[i], s2[j])
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


    pairs = [[s1norm[i[0]], s2norm[i[1]]] for i in indexes]

    # change empty strings to None
    for pair in pairs:
        if pair[0] == '':
            pair[0] = None
        elif pair[1] == '':
            pair[1] = None
        pair = tuple(pair)
    return [pair for pair in pairs if pair[0] is not None]



def nest_list(ls, n):
    '''

        >>> import yapic_io.utils as ut
        >>> t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>>
        >>> res = ut.nest_list(t, 3)
        >>> print(res)
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>>
        >>> res2 = ut.nest_list(t, 4)
        >>> print(res2)
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]


    '''
    indices = list(range(len(ls)))
    start_indices = indices[0::n]
    stop_indices = indices[n::n]

    if len(stop_indices) == len(start_indices)-1:
        stop_indices.append(None)

    return [ls[start_ind:stop_ind] for start_ind, stop_ind in zip(start_indices, stop_indices)]

