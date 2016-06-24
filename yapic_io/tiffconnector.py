import numpy as np
from yapic_io.utils import getFilelistFromDir

class Tiffconnector(object):
    '''
    provides connectors to pixel data source and
    assigned weights for classifier training

    provides methods for getting image templates 

    
    '''
    

    def __init__(self, img_filepath, label_filepath):
        
        self.img_filepath = img_filepath
        self.label_filepath = label_filepath

        self.filenames = None
        

    def __repr__(self):

        infostring = \
            'Connector_tiff object \n' \
            'image filepath: %s \n' \
            'label filepath: %s \n' \
             % (self.img_filepath\
                , self.img_filepath)

        return infostring


    def load_img_filenames(self):
        img_filenames = getFilelistFromDir(self.img_filepath,'.tif')
        print(self.img_filepath)
        print(img_filenames)
        self.filenames = [(filename, ) for filename in img_filenames]



        
