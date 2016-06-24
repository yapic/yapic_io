import numpy as np
from yapic_io.utils import getFilelistFromDir
import yapic_io.image_importers as ip
import logging
import os
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

        self.filenames = None
        self.dimensions =  None

    def __repr__(self):

        infostring = \
            'Connector_tiff object \n' \
            'image filepath: %s \n' \
            'label filepath: %s \n' \
             % (self.img_filepath\
                , self.img_filepath)

        return infostring


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
    
    def scan_dataset_dimensions(self):
        '''
        sets dimensions of the dataset.
        self.dimensions is a list of 4-element-tuples:
        (nr_channels, nr_zslices, nr_x, nr_y)
        each tuple indicates the image dimension of one tiff file
        (respective filenames in self.filenames)

        :returns True if images are found

        '''

        if self.filenames is None:
            logger.warning('dataset empty,no image filenames defined')
            return False

        dims = [ip.get_tiff_image_dimensions(self.img_filepath + '/' + filename[0])\
             for filename in self.filenames] 

        self.dimensions = dims
        return True        
            

            




        
