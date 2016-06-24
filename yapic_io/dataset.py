import numpy as np
from yapic_io.utils import get_template_meshgrid
from functools import lru_cache

class Dataset(object):
    '''
    provides connectors to pixel data source and
    (optionally) assigned weights for classifier training

    provides methods for getting image templates and data 
    augmentation (mini batch) for training

    an image object is loaded into memory, so it should not
    exceed a certain size.

    to handle large images, these should be divided into
    multiple image objects and handled in a training_data class
    (or prediction_data) class
    '''
    

    def __init__(self, pixel_connector):
        
        self.pixel_connector = pixel_connector
        
        self.n_images = pixel_connector.get_image_count()
        


    def __repr__(self):

        infostring = \
            'YAPIC Dataset object \n' +\
            'nr of images: %s \n' % (self.n_images)\
               

        return infostring

    
    @lru_cache(maxsize = 1000)
    def get_img_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''      
        return self.pixel_connector.load_img_dimensions(image_nr)


    #def get_pre_template(image_nr, pos, size):



    def get_template(image_nr, pos, size):
        '''
        returns a recangular subsection of an image with specified size.
        :param image: n dimensional image
        :type image: numpy array
        :param pos: tuple defining the upper left position of the template in n dimensions
        :type pos: tuple
        :param size: tuple defining size of template in all dimensions
        :type size: tuple
        :returns: template as numpy array with same nr of dimensions as image
        '''
        pos = np.array(pos)
        size = np.array(size)


        # if padding == 0:    
        #     return self.pixel_connector.get_template(self, image_nr, pos, size)
        #     #return image[get_template_meshgrid(image.shape, pos, size)]
        
        # pos_p = pos-padding
        # size_p = size + 2*padding
        
        # reflect_sizes = get_padding_size(image.shape, pos_p, size_p)
        # image = np.pad(image, reflect_sizes, mode='reflect')
        # pos_corr = correct_pos_for_padding(pos_p, reflect_sizes)

        # return image[get_template_meshgrid(image.shape, pos_corr, size_p)]    


def get_padding_size(shape, pos, size):
    padding_size = []
    for sh, po, si in zip(shape, pos, size):
        p_l = 0
        p_u = 0
        if po<0:
            p_l = abs(po)
        if po + si > sh:
            p_u = po + si - sh
        padding_size.append((p_l,p_u))        
    return padding_size


def get_dist_to_upper_img_edge(shape, pos):
    return np.array(shape)-np.array(pos)-1


def pos_shift_for_padding(shape, pos, size):
    padding_size = get_padding_size(shape, pos, size)
    dist = get_dist_to_upper_img_edge(shape, pos)

    return dist + 1 - padding_size



    