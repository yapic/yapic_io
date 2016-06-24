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
        print('self.filenames')
        print(self.filenames)
        return len(self.filenames)

    def get_template(self, image_nr, pos, size):

        im = self.load_image(image_nr)
        mesh = get_template_meshgrid(im.shape, pos, size)

        return(im[mesh])


    def get_img_dimensions(self, image_nr):
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
        
    @lru_cache(maxsize = 20)
    def load_image(self, image_nr):
        if not self.is_valid_image_nr(image_nr):
            return None      
        path =  self.img_filepath + '/' + self.filenames[image_nr][0]       
        return ip.import_tiff_image(path)    
    
   


    def is_valid_image_nr(self, image_nr):
        count = self.get_image_count()
        

        error_msg = \
        'wrong image number. image numbers in range 0 to %s' % str(count-1)
        
        if image_nr not in list(range(count)):
            logger.error(error_msg)
            return False

        return True          

    
    def load_img_filenames(self):
        '''
        find all tiff images in specified folder (self.img_filepath)
        '''

        img_filenames = getFilelistFromDir(self.img_filepath,'.tif')
        print(self.img_filepath)
        print(img_filenames)
        filenames = [(filename, ) for filename in img_filenames]
        if len(filenames) == 0:
            self.filenames = None
            return
        self.filenames = filenames
        return True 


    
                          