import os
import numpy as np
import logging
import os
import random
import itertools
logger = logging.getLogger(os.path.basename(__file__))


def get_indices(pos, size):
    '''
    returns all indices for a sub matrix, given a certain n-dimensional position (upper left)
    and n-dimensional size. 


    :param pos: tuple defining the upper left position of the template in n dimensions
    :param size: tuple defining size of template in all dimensions
    :returns: list of indices for all dimensions
    '''
    if len(pos) != len(size):
        error_str = '''nr of dimensions does not fit: pos(%s), 
            size (%s)''' % (pos, size)
        raise ValueError(error_str)


    return [list(range(p, p+s)) for p,s in zip(pos,size)]

def get_template_meshgrid(image_shape, pos, size):
    '''
    returns coordinates for selection of a sub matrix
    , given a certain n-dimensional position (upper left)
    and n-dimensional size. 

    :param shape: tuple defining the image shape
    :param pos: tuple defining the upper left position of the template in n dimensions
    :param size: tuple defining size of template in all dimensions
    :returns: a multidimensional meshgrid defining the sub matrix

    '''

    pos = np.array(pos)
    size = np.array(size)

    # if len(image_shape) != len(size):        
    #     error_str = '''nr of image dimensions (%s) 
    #         and size vector length (%s) do not match'''\
    #         % (len(image_shape), len(size))
    #     raise ValueError(error_str)

    # if len(image_shape) != len(pos):        
    #     error_str = '''nr of image dimensions (%s) 
    #         and pos vector length (%s) do not match'''\
    #         % (len(image_shape), len(pos))
    #     raise ValueError(error_str)    

    
    # if (image_shape < (pos + size)).any():
    #     error_str = '''template out of image bounds: image shape: %s,  
    #         pos: %s, template size: %s''' % (image_shape, pos, size)
    #     raise ValueError(error_str)
    if not is_valid_image_subset(image_shape, pos, size):
        raise ValueError('image subset not valid')


    indices = get_indices(pos, size)
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
        #raise ValueError(error_str)

    if len(image_shape) != len(pos):        
        error_str = '''nr of image dimensions (%s) 
            and pos vector length (%s) do not match'''\
            % (len(image_shape), len(pos))
        logger.error(error_str)
        return False
        #raise ValueError(error_str)    

    
    if (image_shape < (pos + size)).any():
        error_str = '''template out of image bounds: image shape: %s,  
            pos: %s, template size: %s''' % (image_shape, pos, size)
        logger.error(error_str)
        return False
        #raise ValueError(error_str)

    if (pos < 0).any():
        error_str = '''template out of image bounds: image shape: %s,  
            pos: %s, template size: %s''' % (image_shape, pos, size)
        logger.error(error_str)
        return False    
    return True    


def flatten_label_coordinates(label_coordinates):
    return [(label, coor) for label in label_coordinates.keys() for coor in label_coordinates[label]]


def get_max_pos_for_tpl(size, shape):
    return tuple(np.array(shape) - np.array(size))


def get_random_pos_for_coordinate(coor, size, shape):

    if len(coor) != len(size):
        raise ValueError('coor and size must have same lengths. coor: %s, size: %s' \
            % (str(coor), str(size)))

    coor = np.array(coor)
    size = np.array(size)

    
    maxpos = np.array(get_max_pos_for_tpl(size, shape))
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
    computes all possible positions for fetching templates of a given size
    if the templates do not fit perfectly in the image, the last positions 
    are corrected, such that the last and the second-last template would
    have some overlap.
    '''



    shape = np.array(shape)
    size = np.array(size)

    if (size > shape).any():
        raise ValueError('template size is larger than image shape. size: %s, shape: %s' \
            % (str(size), str(shape)))


    nr_tpl = np.ceil(shape/size)
    

    shift_last_tpl = np.zeros(len(shape)).astype(int)
    mod = shape % size #nr of out of bounds pixels for last template
    # print('mod')
    # print(mod)
    # print('size')
    # print(size)
    # print('shift_last_tpl')
    # print(shift_last_tpl)
    

    if mod.any():
        shift_last_tpl[mod!=0] = mod[mod!=0] - size[mod!=0]
    

    pos_per_dim = \
        [list(range(0,imlength,templength)) for imlength, templength in zip(shape,size)]

    
    

    for el,shift in zip(pos_per_dim, shift_last_tpl):   
        el[-1] = el[-1] + shift
        
        
    pos = list(itertools.product(*pos_per_dim))
    

    return pos 








