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
    def image_count(self):
        '''
        :returns: the number of images in the dataset
        '''
        pass

    @abstractmethod
    def get_labelcount_for_im(self, image_nr):
        '''
        returns for each label value the number of labels for this image
        
        label_counts = {
             label_value_1 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_2 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_3 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             ...
        }

        :param image_nr: index of image
        :returns label_counts: dict with counts per label and per image, see example above
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
    def get_template_for_label(self, image_nr, pos_zxy, size_zxy, label_value):
        '''
        returns a 3d zxy boolean matrix where positions of the reuqested label
        are indicated with True. only mapped labelvalues can be requested.

        dimension order: (z,x,y)

        :param image_nr: index of image
        :param pos_zxy: upper left position of subsection (zslice,x,y)
        :type pos_zxy: 4 element tuple of integers (zslice, x, y)
        :param label_value: the id of the label
        :type label_value: integer
        :returns: 3D subsection of labelmatrix as boolean numpy array
        '''    

        pass


    @abstractmethod    
    def put_template(self, pixels, pos_zxy, image_nr, label_value):
        '''
        Puts probabilities (pixels) for a certain label to the data storage.
        These probabilities are prediction values from a classifier (i.e. the
        output of the classier)

        :param pixels: 3D matrix of probability values
        :type pixels: 3D numpy array of floats with shape (z,x,y)
        :param pos_zxy: upper left position of pixels in source image_nr
        :type pos_zxy: tuple of length 3 (z,x,y) with integer values
        :param image_nr: index of image
        :param label_value: the id of the label
        :type label_value: integer
        :returns: bool (True if successful)
        '''


    @abstractmethod
    def image_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)
        '''
        pass

    @abstractmethod
    def get_label_coordinate(self, image_nr, label_value, label_index):
        '''
        returns a czxy coordinate of a specific label (specified by the
        label index) with labelvalue label_value (mapped label value).

        The count of labels for a specific labelvalue can be retrieved by
        
        count = get_labelcount_for_im()

        The label_index must be a value between 0 and count[label_value].
        '''      
    
    # @abstractmethod 
    # def get_label_coordinates(self, image_nr):
    #     ''''
    #     returns label coordinates as dict in following format:
    #     channel has always value 0!! This value is just kept for consitency in 
    #     dimensions with corresponding pixel data 

    #     {
    #         label_nr1 : numpy.array([[c,z,x,y],
    #                                  [c,z,x,y],
    #                                  [c,z,x,y],
    #                                  [c,z,x,y],
    #                                  ...]),
    #         label_nr2 : numpy.array([[c,z,x,y],
    #                                  [c,z,x,y],
    #                                  [c,z,x,y],
    #                                  [c,z,x,y],
    #                                  ...]),
    #         ...
    #     }


    #     :param image_nr: index of image
    #     :returns: dict of label coordinates
    
    #     '''
    #     pass




