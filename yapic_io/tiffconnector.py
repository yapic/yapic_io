import numpy as np
from yapic_io.utils import getFilelistFromDir
import yapic_io.image_importers as ip
import logging
import os
from yapic_io.utils import get_template_meshgrid
from functools import lru_cache

logger = logging.getLogger(os.path.basename(__file__))


class Tiffconnector(object):
    '''
    provides connectors to pixel data source and
    assigned weights for classifier training

    provides methods for getting image templates 

    
    '''
    

    def __init__(self, img_filepath, label_filepath):
        
        self.img_filepath = os.path.normpath(img_filepath)
        self.label_filepath = os.path.normpath(label_filepath)

        self.load_img_filenames()
        self.load_label_filenames()
        self.check_labelmat_dimensions()
        
        

    def __repr__(self):

        infostring = \
            'Connector_tiff object \n' \
            'image filepath: %s \n' \
            'label filepath: %s \n' \
             % (self.img_filepath\
                , self.img_filepath)

        return infostring


    
    

    

    def get_image_count(self):
        
        if self.filenames is None:
            return 0
        #print('self.filenames')
        #print(self.filenames)
        return len(self.filenames)

    def get_template(self, image_nr, pos, size):

        im = self.load_image(image_nr)
        mesh = get_template_meshgrid(im.shape, pos, size)

        return(im[mesh])


    def load_img_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''        

        if not self.is_valid_image_nr(image_nr):
            return False          

        path =  self.img_filepath + '/' + self.filenames[image_nr][0]   
        return ip.get_tiff_image_dimensions(path) 

    def load_labelmat_dimensions(self, image_nr):
        '''
        returns dimensions of the label image.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''        

        if not self.is_valid_image_nr(image_nr):
            return False          

        if self.exists_label_for_img(image_nr):
            path =  self.label_filepath + '/' + self.filenames[image_nr][1]   
            return ip.get_tiff_image_dimensions(path)               
    

    def check_labelmat_dimensions(self):
        '''
        check if label mat dimensions fit to image dimensions, i.e.
        everything identical except nr of channels (label mat always 1)
        '''
        logger.info('labelmat dimensions check')
        for image_nr in list(range(self.get_image_count())):
            im_dim = self.load_img_dimensions(image_nr)
            label_dim = self.load_labelmat_dimensions(image_nr)
            
            if label_dim is None:
                logger.info('check image nr %s: ok (no labelmat found) ', image_nr)
            else:
                if (label_dim[0]  == 1) and (label_dim[1:] == im_dim[1:]):
                    logger.info('check image nr %s: ok ', image_nr)
                else:
                    logger.info('check image nr %s: dims do not match ', image_nr)   
        
    @lru_cache(maxsize = 20)
    def load_image(self, image_nr):
        if not self.is_valid_image_nr(image_nr):
            return None      
        path =  self.img_filepath + '/' + self.filenames[image_nr][0]       
        return ip.import_tiff_image(path)    
    
    def exists_label_for_img(self, image_nr):
        if not self.is_valid_image_nr(image_nr):
            return None      
        if self.filenames[image_nr][1] is None:
            return False
        return True    

    @lru_cache(maxsize = 20)    
    def load_label_matrix(self, image_nr):
        
        if not self.is_valid_image_nr(image_nr):
            return None      

        label_filename = self.filenames[image_nr][1]      
        
        if label_filename is None:
            logger.warning('no label matrix file found for image file %s', str(image_nr))    
            return None
        
        path =  self.label_filepath + '/' + label_filename         
        logger.info('try loading labelmat %s',path)
        return ip.import_tiff_image(path)    
        
    def get_labelvalues_for_im(self, image_nr):
        mat = self.load_label_matrix(image_nr)
        if mat is None:
            return None
        values =  np.unique(mat)
        values = values[values!=0]
        values.sort()
        return list(values)

    def get_label_coordinates(self, image_nr):
        ''''
        returns label coordinates as dict in following format:
        channel has always value 0!! This value is just kept for consitency in 
        dimensions with corresponding pixel data 

        {
            label_nr1 : [(channel, z, x, y), (channel, z, x, y) ...],
            label_nr2 : [(channel, z, x, y), (channel, z, x, y) ...],
            ...
        }


        :param image_nr: index of image
        :returns: dict of label coordinates
    
        '''

        mat = self.load_label_matrix(image_nr)
        labels = self.get_labelvalues_for_im(image_nr)
        if labels is None:
            #if no labelmatrix available
            return None
        
        label_coor = {}
        for label in labels:
            coors = np.array(np.where(mat==label))
            coor_list = [tuple(coors[:,i]) for i in list(range(coors.shape[1]))]
            label_coor[label] = coor_list#np.array(np.where(mat==label))
        
        return label_coor    

    def is_valid_image_nr(self, image_nr):
        count = self.get_image_count()
        

        error_msg = \
        'wrong image number. image numbers in range 0 to %s' % str(count-1)
        
        if image_nr not in list(range(count)):
            logger.error(error_msg)
            return False

        return True          

    
    def load_label_filenames(self, mode='identical'):
        if self.filenames is None:
            return

        if mode == 'identical':
            #if label filenames should match exactly with img filenames
            for name_tuple in self.filenames:
                img_filename = name_tuple[0]   
                label_path =  self.label_filepath + '/' +  img_filename
                if os.path.isfile(label_path): #if label file exists for image file
                    name_tuple[1] = img_filename
    def load_img_filenames(self):
        '''
        find all tiff images in specified folder (self.img_filepath)
        '''

        img_filenames = getFilelistFromDir(self.img_filepath,'.tif')
        
        filenames = [[filename, None] for filename in img_filenames]
        if len(filenames) == 0:
            self.filenames = None
            return
        self.filenames = filenames
        return True 


    
                          