from abc import ABCMeta, abstractmethod

class Connector(metaclass=ABCMeta):
    '''
    Interface to pixel and label data source for classifier 
    training and prediction.

    - Connects to a fixed collection of images and corresponding labels

    - Provides methods for getting sizes of all images

    - Provides methods for getting image subsets and label coordinates.


    Prerequisites for the Image data:

    - Images are 4D datasets with following dimensions: (channel, z, x, y)
    
    - All images must have the same nr of channels but can vary in z,x, and y.
    
    - Time series are not supported. However, time frames can be modeled as
      individual images.  
    
    - Images with lower dimensions can be modeled. E.g. grayscale single plane images
      have the shape (1, 1, width, height). 



    The Connector methods are used by the Dataset class.


    '''
    
    @abstractmethod
    def get_image_count(self):
        '''
        :returns: the number of images in the dataset
        '''
        pass

    @abstractmethod
    def get_template(self, image_nr=None, pos=None, size=None):
        '''
        returns 4D subsection of one image in following dimension order:
        (channel, zslice, x, y)

        :param image_nr: index of image
        :param pos: upper left position of subsection
        :type pos: 4 element tuple of integers (channel, zslice, x, y)
        :returns: 4D subsection of image as numpy array
        '''
        pass

    @abstractmethod
    def load_img_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)
        '''
        pass

    @abstractmethod 
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
        pass




