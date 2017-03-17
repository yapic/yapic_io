import numpy as np

import yapic_io.utils as ut
from functools import lru_cache
import logging
import os
import yapic_io.transformations as trafo
import random
import collections
logger = logging.getLogger(os.path.basename(__file__))

class Dataset(object):
    '''
    provides connectors to pixel data source and
    (optionally) assigned weights for classifier training

    provides methods for getting image templates and data 
    augmentation for training

    pixel data is loaded lazily to allow images of arbitrary size
    pixel data is cached in memory for repeated requests
    
    '''
    def __init__(self, pixel_connector):
        
        self.pixel_connector = pixel_connector
        
        self.n_images = pixel_connector.image_count()
        
        self.label_counts = self.load_label_counts()
        #self.load_label_coordinates() #to be removed
        self.init_label_weights() 

        self.training_template = collections.namedtuple(\
            'TrainingTemplate',['pixels', 'channels', 'weights', 'labels', 'augmentation'])

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
        return self.pixel_connector.image_dimensions(image_nr)

    
    def get_channels(self):
        nr_channels = self.get_img_dimensions(0)[0]
        
        return list(range(nr_channels))

    def get_label_values(self):
        labels = list(self.label_counts.keys())
        labels.sort()  
        return labels    

    #def get_pre_template(image_nr, pos, size):


    def put_prediction_template(self, probmap_tpl, pos_zxy, image_nr, label_value):
        #check if pos and tpl size are 3d
        if not self.image_nr_is_valid(image_nr): 
            raise ValueError('no valid image nr: %s'\
                                % str(image_nr))
        
        
        if not _check_pos_size(pos_zxy, probmap_tpl.shape, 3):
            raise ValueError('pos, size or shape have %s dimensions instead of 3'\
                                % str(len(pos_zxy)))

        #if not self.label_value_is_valid(label_value):
            
            #raise ValueError('label value not found: %s'\
            #                    % str(label_value))    

        
        return self.pixel_connector.put_template(probmap_tpl, pos_zxy, image_nr, label_value)  


    def pick_random_training_template(self, size_zxy, channels, pixel_padding=(0,0,0),\
             equalized=False, rotation_angle=0, shear_angle=0, labels='all', label_region=None):

        if labels == 'all':
            labels = self.get_label_values()
            
        if label_region is None:
        #pick training template wehre it is assured that weights for a specified label
        #are within the template. the specified label is label_region    
            _, coor = self.pick_random_label_coordinate(equalized=equalized)
        else:
            _, coor = self.pick_random_label_coordinate_for_label(label_region)    
        
        img_nr = coor[0]
        #coor_zxy = list(coor)[2:]
        coor_zxy = coor[2:]
        shape_zxy = self.get_img_dimensions(img_nr)[1:]
        pos_zxy = np.array(ut.get_random_pos_for_coordinate(coor_zxy, size_zxy, shape_zxy))

        
        tpl_data = self.get_training_template(img_nr, pos_zxy, size_zxy,\
                channels, labels, pixel_padding=pixel_padding,\
                rotation_angle=rotation_angle, shear_angle=shear_angle)


        return tpl_data

       
    def get_training_template(self, image_nr, pos_zxy, size_zxy, channels, labels,\
            pixel_padding=(0, 0, 0), rotation_angle=0, shear_angle=0):

               
        pixel_tpl =  self.get_multichannel_pixel_template(image_nr, pos_zxy, size_zxy, channels,\
            pixel_padding=pixel_padding, rotation_angle=rotation_angle,\
             shear_angle=shear_angle)#4d pixel template with selected channels in
        #1st dimension

        label_tpl = []



        for label in labels:
            label_tpl.append(self.get_template_for_label(image_nr,\
                    pos_zxy, size_zxy, label,\
                    rotation_angle=rotation_angle, shear_angle=shear_angle))
        label_tpl = np.array(label_tpl) #4d label template with selected labels in
        #1st dimension

        augmentation = {'rotation_angle' : rotation_angle, 'shear_angle' : shear_angle}
        return self.training_template(pixel_tpl, channels, label_tpl, labels, augmentation)

        # return {'pixels' : pixel_tpl,\
        #         'channels' : channels,\
        #         'weights' : label_tpl,\
        #         'labels' : labels}


       
    def get_multichannel_pixel_template(self, image_nr, pos_zxy, size_zxy, channels,\
            pixel_padding=(0, 0, 0), rotation_angle=0, shear_angle=0):

        if not _check_pos_size(pos_zxy, size_zxy,3):
            logger.debug('checked pos size in get_multichannel_pixel_template')
            raise ValueError('pos and size must have length 3. pos_zxy: %s, size_zxy: %s' \
            % (str(pos_zxy), str(size_zxy)))
            

        image_shape_zxy = self.get_img_dimensions(image_nr)[1:]
        if not ut.is_valid_image_subset(image_shape_zxy, pos_zxy, size_zxy):
            raise ValueError('image subset not correct')

        #pos_zxy = np.array(pos_zxy)
        #size_zxy = np.array(size_zxy)
        pixel_padding = np.array(pixel_padding)

        size_padded = size_zxy + 2 * pixel_padding
        pos_padded = pos_zxy - pixel_padding

        pixel_tpl =  []
        for channel in channels:
            
            pixel_tpl.append(self.get_template_singlechannel(image_nr,\
                    tuple(pos_padded), tuple(size_padded), channel,\
                    rotation_angle=rotation_angle, shear_angle=shear_angle))
        pixel_tpl = np.array(pixel_tpl) #4d pixel template with selected channels in
        #1st dimension

        return pixel_tpl




    
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
            raise ValueError('no valid image nr: %s'\
                                % str(image_nr))
        
        
        if not _check_pos_size(pos_zxy, size_zxy, 3):
            raise ValueError('pos, size or shape have %s dimensions instead of 3'\
                                % len(pos_zxy))
        
        shape_zxy = self.get_img_dimensions(image_nr)[1:]    
        #shape_czxy = self.get_img_dimensions(image_nr)
        #pos_czxy = (0,) + tuple(pos_zxy)
        #size_czxy = (1,) + tuple(size_zxy)

        tpl = get_augmented_template(shape_zxy, pos_zxy, size_zxy, \
          self._get_template_for_label_inner,\
          rotation_angle=rotation_angle, shear_angle=shear_angle, reflect=reflect,\
          **{'image_nr' : image_nr, 'label_value' : label_value})    

        return tpl 


    
   
    def _get_template_for_label_inner(self, image_nr=None\
            , pos=None, size=None, label_value=None):
        '''
        returns a 3d weight matrix template for a ceratin label with dimensions zxy.
        '''
        if not self.label_value_is_valid(label_value):
            return False

        pos_zxy = pos
        size_zxy = size

        #shape_zxy = self.get_img_dimensions(image_nr)[1:]
        
        #label_coors = self.label_coordinates[label_value]
        label_weight = self.label_weights[label_value]
        

        boolmat = self.pixel_connector.get_template_for_label(image_nr\
                          , pos_zxy, size_zxy, label_value)
        
        
        weight_mat = np.zeros(boolmat.shape)
        weight_mat[boolmat] = label_weight
        #label_coors_zxy = label_coors[label_coors[:,0]==image_nr,2:]
        

        return weight_mat

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
        
        self.label_weights[label_value] = weight     
        # self.label_weights[label_value] = \
        #     [weight for e in self.label_weights[label_value]]    
        return True
            
    def init_label_weights(self):
        '''
        Inits a dictionary with label weights. All weights are set to 1.
        the self.label_weights dict is complementary to self.label_counts. It defines
        a weight for each label.

        { 
            label_nr1 : weight
            label_nr2 : weight            
        }

        '''
        weight_dict = {}
        label_values =  self.label_counts.keys()
        for label in label_values:
            weight_dict[label] = 1
            #weight_dict[label] = [1 for el in self.label_coordinates[label]]
        self.label_weights = weight_dict
        return True    

    def equalize_label_weights(self):
        '''
        equalizes labels according to their amount.
        less frequent labels are weighted higher than more frequent labels
        '''


        
        labels = self.label_weights.keys()
        total_label_count = dict.fromkeys(labels)

        
        
        for label in labels:
            total_label_count[label] = self.label_counts[label].sum()  
            

        self.label_weights = calc_equalized_label_weights(total_label_count)

            
        return True    

    def load_label_counts(self):
        '''
        returns the cout of each labelvalue for each image as dict


        label_counts = {
             label_value_1 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_2 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_2 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             ...
        }

        '''
        logger.info('start loading label counts...')
        label_counts_raw = [self.pixel_connector.get_labelcount_for_im(im) \
                           for im in range(self.n_images)]
        
        #identify all label_values in dataset
        label_values = []
        for label_count in label_counts_raw:
            if label_count is not None:
                label_values += label_count.keys()
        label_values = set(label_values)
        
        #init empty label_counts dict
        label_counts = {key: np.zeros(self.n_images, dtype='int64') for key in label_values}

        

        for i in range(self.n_images):
            if label_counts_raw[i] is not None:
                for label_value in label_counts_raw[i].keys():
                    label_counts[label_value][i]= label_counts_raw[i][label_value] 
        

        logger.info(label_counts)
        return label_counts    




    def label_value_is_valid(self, label_value):
        '''
        check if label value is part of self.label_coordinates
        '''

        if label_value not in self.label_counts.keys():
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

    
    
           

    def get_nth_label_coordinate_for_label(self, label_value, n):
        '''
        for each labelvalue there exist n labels in the dataset.

        returns the coordinate (image_nr,channel,z,x,y) of the nth label.

        '''
        if not self.label_value_is_valid(label_value):
            raise ValueError('label value %s not valid, possible label values are %s'\
                % (str(label_value),str(self.get_label_values())))

        
        choice = n + 1

        counts = self.label_counts[label_value]
        counts_cs = counts.cumsum()
        total_count = counts_cs[-1]
        if (choice > total_count) or choice < 1:
            raise ValueError(''''choice %s is not valid, only %s labels available
                in the dataset for labelvalue %s''' % (str(choice-1), str(total_count), str(label_value)))
        

        #choice = random.randint(0,total_count-1)

        
        image_nr = (counts_cs >= choice).nonzero()[0][0]
        counts_cs_shifted = np.insert(counts_cs,0,[0])[:-1] #shift cumulated counts by 1
        
        label_index = choice - counts_cs_shifted[image_nr] - 1
        
        coor_czxy = self.pixel_connector.get_label_coordinate(image_nr\
                                                       , label_value\
                                                       , label_index)
        coor_iczxy = np.insert(coor_czxy,0,image_nr)
        return coor_iczxy



    def pick_random_label_coordinate_for_label(self, label_value):
        '''
        returns a rondomly chosen label coordinate and the label value for a givel label value:
        
        (label_value, (img_nr,channel,z,x,y))

        channel is always zero!!

        :param equalized: If true, less frequent label_values are picked with same probability as frequent label_values
        :type equalized: bool 
        '''
        
        if not self.label_value_is_valid(label_value):
            raise ValueError('label value %s not valid, possible label values are %s'\
                % (str(label_value),str(self.get_label_values())))

        counts = self.label_counts[label_value]
        total_count = counts.sum()

        if total_count < 1: #if no labels of that value
            raise ValueError('no labels of value %s existing' % str(label_value))
        choice = random.randint(0,total_count-1)

        return (label_value, self.get_nth_label_coordinate_for_label(label_value, choice))

    
    
            

    def pick_random_label_coordinate(self, equalized=False):
        '''
        returns a randomly chosen label coordinate and the label value:
        
        (label_value, (img_nr,channel,z,x,y))

        channel is always zero!!

        :param equalized: If true, less frequent label_values are picked with same probability as frequent label_values
        :type equalized: bool 
        '''
        labels = self.get_label_values()
        if equalized:
            label_sel = random.choice(labels)    
            return self.pick_random_label_coordinate_for_label(label_sel)

        else:
            label_values = []
            total_counts = []
            for label_value in self.label_counts.keys():
                label_values.append(label_value)
                total_counts.append(self.label_counts[label_value].sum())

            #probabilities for each labelvalue
            total_counts_norm = np.array(total_counts)/sum(total_counts)
            total_counts_norm_cs = total_counts_norm.cumsum()
            label_values = np.array(label_values)
            
            #pick a labelvalue according to the labelvalue probability
            random_nr = random.uniform(0,1)
            chosen_label = label_values[(random_nr <= total_counts_norm_cs).nonzero()[0][0]]
            
            return self.pick_random_label_coordinate_for_label(chosen_label)

            


def get_label_values(self):
    return set(self.label_counts.keys())



        
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





# def label2mask(image_shape, pos, size, label_coors, weights):
#     '''
#     transform label coordinate to mask of given pos and size inside image
#     '''

#     if not ut.is_valid_image_subset(image_shape, pos, size):
#         raise ValueError('image subset (labels) not valid')
#     msk = np.zeros(size)
#     #label_coors_corr = np.array(label_coors) - np.array(pos) + 1

    
#     #pos = np.array(pos)
#     #size = np.array(size)
    

#     if label_coors.size == 0:
#         return msk

#     coor_shifted = label_coors - pos
#     coor_mask = np.logical_and((coor_shifted > -1).all(axis=1),
#                                (size - coor_shifted > 0).all(axis=1))
#     indices1d = np.ravel_multi_index(coor_shifted[coor_mask].T, size)
#     msk.flat[indices1d] = weights[coor_mask]

#     return msk    





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
    :returns dict with equalized weights for each label value
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


def get_augmented_template(shape, pos, size,\
        get_template_func, rotation_angle=0, shear_angle=0,\
        reflect=True, **kwargs):
    '''
    fixme: morph template works only in 2d.
    morphing has to be applied slice by slice
    '''
    
    

    if (rotation_angle == 0) and (shear_angle == 0):
        return get_template_with_reflection(shape,\
             pos, size, get_template_func, reflect=reflect, **kwargs)
    
    if (size[-2]) == 1 and (size[-1] == 1):
        #if the requested template is only of size 1 in x and y,
        #augmentation can be omitted, since rotation always
        #occurs around the center axis. 
        return get_template_with_reflection(shape,\
             pos, size, get_template_func, reflect=reflect, **kwargs)


    size = np.array(size)
    pos = np.array(pos)
    
    size_new = size * 3 #triple template size if morphing takes place
    pos_new = pos-size    
    tpl_large = get_template_with_reflection(shape, pos_new, size_new, get_template_func, reflect=reflect, **kwargs)
    tpl_large_morphed = trafo.warp_image_2d_stack(tpl_large, rotation_angle, shear_angle)
    
    mesh = ut.get_template_meshgrid(tpl_large_morphed.shape, size, size)

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
        logger.debug('requested template out of bounds')
        logger.debug('image will be extended with reflection')

    
    #get transient template
    transient_tpl = get_template_func( \
        pos=tuple(pos_transient), size=tuple(size_transient), **kwargs)


    

    #pad transient template with reflection
    transient_tpl_pad = np.pad(transient_tpl, pad_size, mode='symmetric')        

    mesh = ut.get_template_meshgrid(transient_tpl_pad.shape,\
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

