import numpy as np
from yapic_io.utils import get_template_meshgrid, is_valid_image_subset
from functools import lru_cache
import logging
import os
import yapic_io.transformations as trafo
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
        
        self.load_label_coordinates()
        self.init_label_weights()

    def __repr__(self):

        infostring = \
            'YAPIC Dataset object \n' +\
            'nr of images: %s \n' % (self.n_images)\
               

        return infostring

    
    def equalize_label_weights(self):
        '''
        equalizes labels according to their amount.
        less frequent labels are weighted higher than more frequent labels
        '''

        label_n = {}
        labels = self.label_weights.keys()
        nn = 0# total nr of labels
        for label in labels:
            n = len(self.label_weights[label])
            label_n[label] = n #nr of labels for that label value
            

       
        eq_weights = calc_equalized_label_weights(label_n)

        for label in labels:
            self.set_weight_for_label(eq_weights[label], label)
        return True    



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


    # def get_template(self, image_nr, pos_zxy, size_zxy, channels, labels):

    #     pixel_tpl =  []
    #     for channel in channels:
    #         pixel_tpl.append(self.get_template_singlechannel(image_nr, \
    #                 pos_zxy, size_zxy, channel))
    #     pixel_tpl = np.array(pixel_tpl)    
                



    @lru_cache(maxsize = 1000)
    def get_template_singlechannel(self, image_nr\
           , pos_zxy, size_zxy, channel, reflect=True,\
            rotation_angle=0, shear_angle=0):
        '''
        returns a recangular subsection of an image with specified size.
        if requested template is out of bounds, values will be added by reflection

        :param image_nr: image index
        :type image: int
        :param pos: tuple defining the upper left position of the template in 3 dimensions zxy
        :type pos: tuple
        :param size: tuple defining size of template in 3 dimensions zxy
        :type size: tuple
        :returns: 3d template as numpy array with dimensions zxy
        '''
        
        if not self.image_nr_is_valid(image_nr): 
            return False

        
        if not _check_pos_size(pos_zxy, size_zxy, 3):
            return False

        pos_czxy = np.array([channel] + list(pos_zxy))
        size_czxy = np.array([1] + list(size_zxy)) # size in channel dimension is set to 1
        #to only select a single channel

        shape_czxy = self.get_img_dimensions(image_nr)
        #shape_zxy = self.get_img_dimensions(image_nr)[1:]
        #shape[0] = 1 #set nr of channels to 1

        # tpl = get_template_with_reflection(shape_czxy, pos_czxy, size_czxy,\
        #      self.pixel_connector.get_template, reflect=True\
        #      , **{'image_nr' : image_nr})
        
        tpl = get_augmented_template(shape_czxy, pos_czxy, size_czxy, \
          self.pixel_connector.get_template,\
          rotation_angle=rotation_angle, shear_angle=shear_angle, reflect=reflect,\
          **{'image_nr' : image_nr})

        # tpl = get_template_with_reflection(shape_czxy, pos_czxy, size_czxy,\
        #      self.pixel_connector.get_template, reflect=True\
        #      , **{'image_nr' : image_nr})

        return np.squeeze(tpl, axis=(0,))

        
    def get_template_for_label(self, image_nr\
           , pos_zxy, size_zxy, label_value, reflect=True,\
           rotation_angle=0, shear_angle=0):


        '''
        returns a recangular subsection of label weights with specified size.
        if requested template is out of bounds, values will be added by reflection

        :param image_nr: image index
        :type image: int
        :param pos_zxy: tuple defining the upper left position of the template in 3 dimensions zxy
        :type pos_zxy: tuple
        :param size_zxy: tuple defining size of template in 3 dimensions zxy
        :type size: tuple
        :param label_value: label identifier
        :type label_value: int
        :returns: 3d template of label weights as numpy array with dimensions zxy
        '''

        if not self.image_nr_is_valid(image_nr): 
            return False
        shape_zxy = self.get_img_dimensions(image_nr)[1:]     
        
        if not _check_pos_size(pos_zxy, size_zxy, 3):
            return False
        
        tpl = get_augmented_template(shape_zxy, pos_zxy, size_zxy, \
          self._get_template_for_label_inner,\
          rotation_angle=rotation_angle, shear_angle=shear_angle, reflect=reflect,\
          **{'image_nr' : image_nr, 'label_value' : label_value})    

        # tpl = get_template_with_reflection(shape_zxy, pos_zxy, size_zxy,\
        #      self._get_template_for_label_inner, reflect=True\
        #      , **{'image_nr' : image_nr, 'label_value' : label_value})

        return np.squeeze(tpl, axis=(0,))     



            


    def _get_template_for_label_inner(self, image_nr=None\
            , pos=None, size=None, label_value=None):
        '''
        returns a 3d weight matrix template for a ceratin label with dimensions zxy.
        '''
        if not self.label_value_is_valid(label_value):
            return False

        pos_zxy = pos
        size_zxy = size

        shape_zxy = self.get_img_dimensions(image_nr)[1:]
        

        label_coors = self.label_coordinates[label_value]
        label_weights = self.label_weights[label_value]
        
        label_coors_zxy = []
        weights = []
        for coor, weight in zip(label_coors, label_weights):
            if coor[0] == image_nr:
                label_coors_zxy.append(coor[2:])
                weights.append(weight)
        # label_coors_zxy = \
        #     [coor[2:] for coor in label_coors if coor[0] == image_nr]
        

        print('labelcoors_zxy')
        print(label_coors_zxy) 
        print('shape')
        print(shape_zxy) 
        print('pos')
        print(pos_zxy) 
        print('size')
        print(size_zxy) 
        return label2mask(shape_zxy, pos_zxy, size_zxy, label_coors_zxy, weights)

    def set_weight_for_label(self, weight, label_value):
        '''
        sets the same weight for all labels of label_value
        in self.label_weights
        :param weight: weight value
        :param label_value: label 

        '''
        if not self.label_value_is_valid(label_value):
            logger.warning(\
                'could not set label weight for label value %s'\
                , str(label_value))
            return False
        self.label_weights[label_value] = \
            [weight for e in self.label_weights[label_value]]    
        return True
            
    def init_label_weights(self):
        '''
        Inits a dictionary with label weights. All weights are set to 1.
        the self.label_weights dict is complementary to self.label_coordinates. It defines
        a weight for each coordinate.

        { 
            label_nr1 : [
                            weight,
                            weight,
                            weight,
                             ...],
            label_nr2 : [
                            weight,,
                            weight,
                            weight,
                             ...],                 
        }

        '''
        weight_dict = {}
        label_values =  self.label_coordinates.keys()
        for label in label_values:
            weight_dict[label] = [1 for el in self.label_coordinates[label]]
        self.label_weights = weight_dict
        return True    

    

    def load_label_coordinates(self):
        '''
        imports labale coodinates with the connector objects and stores them in 
        self.label_coordinates in following dictionary format:

        { 
            label_nr1 : [
                            (img_nr, channel, z, x, y),
                            (img_nr, channel, z, x, y),
                            (img_nr, channel, z, x, y),
                             ...],
            label_nr2 : [
                            (img_nr, channel, z, x, y),
                            (img_nr, channel, z, x, y),
                            (img_nr, channel, z, x, y),
                             ...],                 
        }

        channel has always value 0!! This value is just kept for consitency in 
        dimensions with corresponding pixel data 

        '''
        
        labels = {}
        
        for image_nr in list(range(self.n_images)):
            
            label_coor = self.pixel_connector.get_label_coordinates(image_nr)
            if label_coor is not None:
                label_coor_5d = label_coordinates_to_5d(label_coor, image_nr)
                if not self.label_coordinates_is_valid(label_coor_5d):
                    logger.warning('failed to load label coordinates because of non valid data')
                    return False

                for key in label_coor_5d.keys():
                    if key not in labels.keys():
                        labels[key] = label_coor_5d[key]
                    else:
                        labels[key] = labels[key] + label_coor_5d[key]

        self.label_coordinates = labels                 
        return True

    def label_value_is_valid(self, label_value):
        '''
        check if label value is part of self.label_coordinates
        '''

        if label_value not in self.label_coordinates.keys():
            logger.warning(\
                'Label value not found %s',\
                 str(label_value))
            return False
        return True   


    def image_nr_is_valid(self, image_nr):
        '''
        check if image_nr is between 0 and self.n_images
        '''

        if (image_nr >= self.n_images) or (image_nr < 0):
            logger.error(\
                'Wrong image number. image numbers in range 0 to %s',\
                 str(self.n_images-1))
            return False
        return True    

    def label_coordinates_is_valid(self, label_coordinates):
        '''
        check if label coordinate data meets following requirements:
        
        z,x,y within image size
        channel = 0 (for label data, always only one channel)
        image_nr between 0 and n_images-1
        no duplicate positions within and across labels


        label_cooridnates = 
        {
                label_nr1 : [(image_nr, channel, z, x, y), (image_nr, channel, z, x, y) ...],
                label_nr2 : [(image_nr, channel, z, x, y), (image_nr, channel, z, x, y) ...],
                ...
            }

        
        '''
        
        #check for duplicates
        coor_flat = []
        for key in label_coordinates.keys():
            coor_flat = coor_flat + label_coordinates[key]

        if len(set(coor_flat)) != len(coor_flat):
            #if threre are duplicate postitions
            logger.warning('duplicate label positions detected')
            return False

        for coor in coor_flat:
            #check for correct nr of dimensions    
            if len(coor) != 5:
                logger.warning(\
                    'number of coordinate dimensions must be 5,' +\
                    ' but is at least one time %s, coordinate: %s',\
                     str(len(coor)), str(coor))
                return False
            if (coor[0] < 0) or (coor[0] >= self.n_images):

                logger.warning(\
                    'image number in label coordinates not valid:' +\
                    ' is %s, but must be between 0 and %s',\
                     str(coor[0]), str(self.n_images-1))
                return False
            
            #image dimensions in z,x and y
            dim_zxy = self.get_img_dimensions(coor[0])[1:]
            coor_zxy = coor[2:]

            #check if coordinates are within image zxy range
            if (np.array(coor_zxy) > np.array(dim_zxy)).any():
                logger.warning(\
                    'label coordinate out of image bounds:' +\
                    ' coordinate: %s, image bounds: %s',\
                     str(coor_zxy), str(dim_zxy))
                return False

            if  (np.array(coor_zxy) < 0).any():
                logger.warning(\
                    'label coordinate below zero:' +\
                    ' coordinate: %s, image bounds: %s',\
                     str(coor_zxy), str(dim_zxy))
                return False 
            
            if coor[1] != 0:
                logger.warning(\
                    'nr of channels MUST be 0 for label coordinates, but is %s' +\
                    ' coordinate: %s',\
                     str(coor[1]), str(coor))
                return False

        return True    


           




def label_coordinates_to_5d(label_dict, image_nr):
    '''
    add image_nr as first dimension to coordinates
    '''
    
    label_dict = dict(label_dict) #copy
    for key in label_dict.keys():
        label_dict[key] =\
             [(image_nr, coor[0], coor[1], coor[2], coor[3]) for coor in label_dict[key]]

    return label_dict         
        
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





def label2mask(image_shape, pos, size, label_coors, weights):
    '''
    transform label coordinate to mask of given pos and size inside image
    '''

    if not is_valid_image_subset(image_shape, pos, size):
        raise ValueError('image subset (labels) not valid')
    msk = np.zeros(size)
    #label_coors_corr = np.array(label_coors) - np.array(pos) + 1



    for coor, weight in zip(label_coors, weights):
        coor_shifted = np.array(coor) - np.array(pos)
        #print(coor_shifted)
        #msk[coor[0],coor[1],coor[2]] = label_value
        #print(coor)
        #print(msk)
        if (coor_shifted >= 0).all() and (coor_shifted < np.array(size)).all():
            msk[tuple(coor_shifted)] = weight
    return msk    




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


def calc_equalized_label_weights(label_n):
    '''
    :param label_n: dict with label numbers where keys are label_values
    :returns dict with equalized weights for eahc label value
    label_n = {
            label_value_1 : n_labels,
            label_value_2 : n_labels,
            ...
            }
    '''
    nn = 0
    labels = label_n.keys()
    for label in labels:
        nn += label_n[label] 
    
    weight_total_per_labelvalue = float(nn)/float(len(labels))

    #equalize
    eq_weight = {}
    eq_weight_total = 0
    for label in labels:
        eq_weight[label] = \
         weight_total_per_labelvalue/float(label_n[label])
        eq_weight_total += eq_weight[label] 

    #normalize
    for label in labels:
        eq_weight[label] = eq_weight[label]/eq_weight_total

    return eq_weight     


def get_augmented_template(shape, pos, size, \
        get_template_func, rotation_angle=0, shear_angle=0,\
        reflect=True, **kwargs):

    if (rotation_angle == 0) and (shear_angle == 0):
        return get_template_with_reflection(shape, pos, size, get_template_func, reflect=reflect, **kwargs)
    
    size = np.array(size)
    pos = np.array(pos)
    
    size_new = size * 3
    pos_new = pos-size    
    tpl_large = get_template_with_reflection(shape, pos_new, size_new, get_template_func, reflect=reflect, **kwargs)
    tpl_large_morphed = trafo.warp_image_2d(tpl_large, rotation_angle, shear_angle)
    
    mesh = get_template_meshgrid(tpl_large_morphed.shape, size, size)

    return tpl_large_morphed[mesh]


def get_template_with_reflection(shape, pos, \
        size, get_template_func, reflect=True, **kwargs):
     

    pos_transient, size_transient, pos_inside_transient, pad_size = \
                calc_inner_template_size(shape, pos, size)

    if is_padding(pad_size) and not reflect:
        #if image has to be padded to get the template
        logger.error('requested template out of bounds')
        return False
    
    if is_padding(pad_size) and reflect:
        #if image has to be padded to get the template and reflection mode is on
        logger.info('requested template out of bounds')
        logger.info('image will be extended with reflection')

    
    #get transient template
    transient_tpl = get_template_func( \
        pos=pos_transient, size=size_transient, **kwargs)


    

    #pad transient template with reflection
    transient_tpl_pad = np.pad(transient_tpl, pad_size, mode='symmetric')        

    mesh = get_template_meshgrid(transient_tpl_pad.shape,\
                     pos_inside_transient, size)

    return transient_tpl_pad[mesh]



def _check_pos_size(pos, size, nr_dim):
           
    error_msg = \
        'Wrong number of image dimensions.' +  \
        'Nr of dimensions MUST be 3 (nr_zslices, nr_x, nr_y), but' + \
        'is  %s for size and %s for pos' % (str(len(size)), str(len(pos)))    

    if (len(pos) != nr_dim) or (len(size) != nr_dim):
        logger.error(error_msg)
        return False
    return True    

