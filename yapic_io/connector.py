from abc import ABCMeta, abstractmethod

class Connector(metaclass=ABCMeta):
    '''
    Interface to pixel and label data source for classifier 
    training and prediction.


    -
    -Provides methods for getting image subsets and label coordinates.


    The Connector methods are used by the Dataset class.


    '''
    
    @abstractmethod
    def get_image_count(self):
        '''
        returns the number of images in the dataset
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




