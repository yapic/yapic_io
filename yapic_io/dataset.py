import numpy as np
from yapic_io.utils import get_template_meshgrid
from functools import lru_cache
import logging
import os
logger = logging.getLogger(os.path.basename(__file__))

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


    @lru_cache(maxsize = 1000)
    def get_template_singlechannel(self, image_nr, pos_zxy, size_zxy, channel, reflect=True):
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
        
        if image_nr >= self.n_images:
            error_msg = \
                'Wrong image number. image numbers in range 0 to %s' % str(self.n_images-1)
            logger.error(error_msg)
            return False

        
        error_msg = \
            'Wrong number of image dimensions.' +  \
            'Nr of dimensions MUST be 3 (nr_zslices, nr_x, nr_y), but' + \
            'is  %s for size and %s for pos' % (str(len(size_zxy)), str(len(pos_zxy)))    

        if (len(pos_zxy) != 3) or (len(size_zxy) != 3):
            logger.error(error_msg)
            return False    

        pos_czxy = np.array([channel] + list(pos_zxy))
        size_czxy = np.array([1] + list(size_zxy)) # size in channel dimension is set to 1
        #to only select a single channel

        shape_czxy = self.get_img_dimensions(image_nr)
        #shape[0] = 1 #set nr of channels to 1

        pos_transient, size_transient, pos_inside_transient, pad_size = \
                calc_inner_template_size(shape_czxy, pos_czxy, size_czxy)

        if is_padding(pad_size) and reflect:
            #if image has to be padded to get the template
            logger.error('requested template out of bounds')
            return False
        
        if is_padding(pad_size) and reflect:
            #if image has to be padded to get the template and reflection mode is on
            logger.info('requested template out of bounds')
            logger.info('image will be extended with reflection')

        #add channel dimension        
        #pos_transient = tuple([channel] + list(pos_transient))
        #size_transient = tuple([1] + list(size_transient))

        #get transient template
        transient_tpl = self.pixel_connector.get_template( \
            image_nr, pos_transient, size_transient)


        

        #pad transient template with reflection
        transient_tpl_pad = np.pad(transient_tpl, pad_size, mode='symmetric')        

        mesh = get_template_meshgrid(transient_tpl_pad.shape,\
                         pos_inside_transient, size_czxy)

        return transient_tpl_pad[mesh]




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
    '''
    [(x_lower, x_upper), (y_lower,_y_upper)]
    '''
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


def is_padding(padding_size):
    '''
    check if are all zero (False) or at least one is not zero (True)
    '''
    for dim in padding_size:
        if np.array(dim).any(): # if any nonzero element
            return True
    return False        


def calc_inner_template_size(shape, pos, size):
    '''
    if a requested template is out of bounds, this function calculates 
    a transient template position size and pos. The transient template has to be padded in a
    later step to extend the edges for delivering the originally requested out of
    bounds template. The padding sizes for this step are also delivered.
    size_out and pos_out that can be used in a second step with padding.
    pos_tpl defines the position in the transient template for cuuting out the originally
    requested template.

    :param shape: shape of full size original image
    :param pos: upper left position of template in full size original image
    :param size: size of template 
    :returns pos_out, size_out, pos_tpl, padding_sizes

    pos_out is the position inside the full size original image for the transient template.  
    (more explanation needed)
    '''

    shape = np.array(shape)
    pos = np.array(pos)
    size = np.array(size)

    padding_sizes = get_padding_size(shape, pos, size)

    
    padding_upper = np.array([e[1] for e in padding_sizes])
    padding_lower = np.array([e[0] for e in padding_sizes])

    shift_1 = padding_lower
    shift_2 = shape - pos - padding_upper
    shift_2[shift_2 > 0] = 0

    shift = shift_1 + shift_2
    pos_tpl = -shift + padding_lower
    pos_out = pos + shift

    dist_lu_s = shape - pos - shift
    size_new_1 = np.vstack([size,dist_lu_s]).min(axis=0)
    pos_r = pos.copy()
    pos_r[pos>0]=0
    size_inmat = size + pos_r
    
    size_new_2 = np.vstack([padding_lower,size_inmat]).max(axis=0)
    size_out = np.vstack([size_new_1,size_new_2]).min(axis=0)

    return tuple(pos_out), tuple(size_out), tuple(pos_tpl), padding_sizes

    



   


def get_dist_to_upper_img_edge(shape, pos):
    return np.array(shape)-np.array(pos)-1


def pos_shift_for_padding(shape, pos, size):
    padding_size = get_padding_size(shape, pos, size)
    dist = get_dist_to_upper_img_edge(shape, pos)

    return dist + 1 - padding_size



    