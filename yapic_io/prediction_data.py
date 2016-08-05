import yapic_io.utils as ut
import numpy as np
from bisect import insort_left
import os
import logging
logger = logging.getLogger(os.path.basename(__file__))

class PredictionData(object):
    '''
    List-like data interface for classification with neural networks.

    - Provides get_pixels() method for getting pixel templates from 
      4D images (channels, z, x,y) that fit into your neural network 
      input layer.

    - Pixel data is loaded lazily to support arbitrary large image datasets.

    - Provides put_probmap_data() method to transfer classification results of your
      neural network back to the data source.
    
    - Flexible data binding through the Dataset object.


        >>> from yapic_io.factories import make_tiff_interface
        >>>
        >>> #define data loacations
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> tpl_size = (1,5,4) # size of network output layer in zxy
        >>> padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
        >>>
        >>> # make minibatch mb and prediction interface p with TiffConnector binding
        >>> _, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size, padding_zxy=padding) 
        >>> len(p) #give total the number of templates that cover the whole bound tiff files 
        510
        >>>
        >>> #classify the whole bound dataset
        >>> counter = 0 #needed for mock data
        >>> for item in p:
        ...     pixels_for_classifier = item.get_pixels() #input for classifier
        ...     mock_classifier_result = np.ones(tpl_size) * counter #classifier output
        ...     #pass classifier results for each class to data source
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=91)
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=109)
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=150)
        ...     counter += 1
        >>>

    
    '''
    

    def __init__(self, dataset, size_zxy, padding_zxy=(0,0,0)):

        '''
        :param dataset: Dataset object for data binding of 
                        image pixel data and classification results.
        :type dataset: Dataset
        :param size_zxy: zxy_size of the neural network output layer.
        :type size_zxy: tuple of 3 integers (z,x,y)
        :param padding_zxy: The padding of the network's input layer
                            relative to its output layer.
        :type padding_zxy: Tuple of 3 integers (z,x,y).
        '''

        self._dataset = dataset
        

        self._size_zxy = None
        self.set_tpl_size_zxy(size_zxy)
        self._padding_zxy = None
        self.set_padding_zxy(padding_zxy)
        self._channels = self._dataset.get_channels() #imports all available channels by default
        self._labels = self._dataset.get_label_values() #imports all available labels by default
        
        #template location info will be set by the __getitem__ method
        self._pos_zxy = None 
        self._image_nr = None 

        #a list of all possible template positions 
        #[(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        self._tpl_pos_all = self._compute_pos_zxy()


        



        
        
        
    def __len__(self):
        '''
        Implements list-like operations.
        '''
        return len(self._tpl_pos_all)
        


    def __getitem__(self, position):
        '''
        Implements list-like operations of element selection and slicing.
        '''

        image_nr, pos_zxy = self._tpl_pos_all[position]
        
        self._image_nr = image_nr
        self._pos_zxy = pos_zxy
        
        return self
       

    def get_tpl_size_zxy(self):
        '''
        Returns the size in (z,x,y) of the template. Should match the size in (z,x,y)
        of the network's output layer.
        '''
        return self._size_zxy

    def set_tpl_size_zxy(self, size_zxy):
        '''
        Sets the size in (z,x,y) of the template. Should match the size in (z,x,y)
        of the network's output layer.

        Must not be larger than the smallest image of the dataset!!
        '''
        
        if len(size_zxy) != 3: 
            raise ValueError(\
                '''no valid size for probmap template: 
                   shape is %s, len of shape should be 3: (z,x,y)'''\
                                % str(size_zxy))

        self._size_zxy = size_zxy

        #a list of all possible template positions 
        #[(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        self._tpl_pos_all = self._compute_pos_zxy()   


    def get_padding_zxy(self):
        '''
        Returns the padding size in (z,x,y), i.e. the padding of the network's
        input layer compared to its output layer.

        Padding size has to be defined by the user according to the used network
        structure.

        '''     
        return self._padding_zxy

    def set_padding_zxy(self, padding_zxy):
        '''
        Sets the padding size in (z,x,y), i.e. the padding of the network's
        input layer compared to its output layer.

        Padding size has to be defined according to the used network
        structure.

        Example: If the output layer's size is (1,6,4) and the input layer's 
                 size is (1,8,6), then the padding in xyz is (0,1,1), i.e 
                 the input layer grows 0 pixels in z, and 1 pixel in
                 in x and y, compared to the output layer. (The growth of
                 1 pixel at both ends results in a matrix that is 2 pixels larger
                 than the output layer).
                
                >>> import numpy as np
                >>> from pprint import pprint
                >>> 
                >>> output_layer = np.zeros((1,6,4))
                >>> input_layer = np.zeros((1,8,6)) #padded by (0,1,1) compared to output_layer
                >>> 
                >>> pprint(output_layer)
                array([[[ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.]]])
                >>> pprint(input_layer)
                array([[[ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.]]])
                   
        '''     
        if len(padding_zxy) != 3: 
            raise ValueError(\
                '''no valid dimension for padding: 
                   shape is %s, len of shape should be 3: (z,x,y)'''\
                                % str(padding_zxy.shape))
        self._padding_zxy = padding_zxy    

    
    def get_channels(self):
        '''
        Returns the selected image channels.
        Per default, all available channels are selected.

        The order of the channels list defines the order of the
        channels layer in the pixels (accessed with get_pixels()).
        Pixels have the dimensionality (channels,z,x,y)
        
        >>> from yapic_io.factories import make_tiff_interface
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> tpl_size = (1,5,4)
        >>> # make minibatch mb and prediction interface p: upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size) #upon object initialization all available image channels are set
        >>>
        >>>
        >>> p.get_channels() #we have 3 channels in the prediction interface
        [0, 1, 2]
        >>> 
        >>> p[0].get_pixels().shape #accordingly, we have 3 channels in the 4D pixels template (c,z,x,y)
        (3, 1, 5, 4)
        >>> 
        '''
        return self._channels    

    def remove_channel(self, channel):
        '''
        Removes a pixel channel from the selection.
        :param channel: channel
        :returns: bool, True if channel was removed from selection

        >>> from yapic_io.factories import make_tiff_interface
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> tpl_size = (1,5,4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size)
        >>>
        >>>
        >>> p.get_channels() #we have 3 channels
        [0, 1, 2]
        >>> p[0].get_pixels().shape #accordingly, we have 3 channels in the 4D pixels template (c,z,x,y)
        (3, 1, 5, 4)
        >>>
        >>> p.remove_channel(1) #remove channel 1
        True
        >>> p.get_channels() #only 2 channels selected
        [0, 2]
        >>> p[0].get_pixels().shape #only 2 channels left in the pixel template
        (2, 1, 5, 4)

        '''    
        if channel not in self._channels:
            raise ValueError('not possible to remove channel %s from channel selection %s'\
                % (str(channel), str(self._channels)))
        self._channels.remove(channel)
        return True    

    
    def add_channel(self, channel):
        '''
        Adds a pixel-channel to the selection.


        '''
        if channel in self._channels:
            logger.warning('channel already selected %s', channel)
            return False
        if channel not in self._dataset.get_channels():
            raise ValueError('not possible to add channel %s from dataset channels %s'\
                % (str(channel), str(self._dataset.get_channels())))

        insort_left(self._channels,channel)    
        return True
        
    
    def get_labels(self):
        '''
        Returns the selected labels (i.e. classes for prediction).
        Per default, all available labels are selected.

        The order of the labels list defines the order of the
        labels layer in the probability map (i.e. the classification result 
        matrix) that can be exported with put_probmap_data().
        
        '''

        return self._labels


    def remove_label(self, label):
        '''
        Removes a label class from the selection.

        :param label: label 
        :returns: bool, True if label was removed from selection

        >>> from yapic_io.factories import make_tiff_interface
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> tpl_size = (1,5,4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size)
        >>>
        >>>
        >>> p.get_labels() #we have 3 label classes
        [91, 109, 150]
        >>> p.remove_label(109) #remove label class 109
        True
        >>> p.get_labels() #only 2 classes remain selected
        [91, 150]
        

        '''    
        if label not in self._labels:
            raise ValueError('not possible to remove label %s from label selection %s'\
                % (str(label), str(self._labels)))
        self._labels.remove(label)
        return True        


    def add_label(self, label):
        '''
        Adds a label class to the selection.

        >>> from yapic_io.factories import make_tiff_interface
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> tpl_size = (1,5,4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size)
        >>>
        >>>
        >>> p.get_labels() #we have 3 label classes
        [91, 109, 150]
        >>> p.remove_label(109) #remove label class 109
        True
        >>> p.get_labels() #only 2 classes remain selected
        [91, 150]
        >>> p.remove_label(91) #remove label class 91
        True
        >>> p.get_labels() #only 1 class remains selected
        [150]
        >>> p.add_label(91)
        True
        >>> p.get_labels() 
        [91, 150]
        >>> p.add_label(109)
        True
        >>> p.get_labels() 
        [91, 109, 150]


        '''
        if label in self._labels:
            logger.warning('label class already selected %s', label)
            return False
        if label not in self._dataset.get_label_values():
            raise ValueError('not possible to add label class %s from dataset label classes %s'\
                % (str(label), str(self._dataset.get_label_values())))

        insort_left(self._labels,label)    
        return True    
        
    def put_probmap_data(self, probmap_data):
        '''
        Put classification results to the data source.

        The order of the labels list (accesed with self.get_labels())defines 
        the order of the labels layer in the probability map.

        To pass 3D probmaps for a certain label, use put_probmap_data_for_label() 

        :param probmap_data: 4D matrix with shape (nr_labels, z,x,y).
        :type probmap_data: numpy.array.
        :returns: bool (True if successful).

        >>> from yapic_io.factories import make_tiff_interface
        >>> from pprint import pprint
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = 'yapic_io/test_data/tmp/'
        >>> 
        >>> #define size of network output layer
        >>> tpl_size = (1,5,4)
        >>> # make minibatch mb and prediction interface p
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size)
        >>> #upon object initialization all available classes are set
        >>> p.get_labels()
        [91, 109, 150]
        >>>
        >>> dummy_predictions_class_91 = np.ones(tpl_size)*0.1 #mocking some prediction data
        >>> dummy_predictions_class_109 = np.ones(tpl_size)*0.2
        >>> dummy_predictions_class_150 = np.ones(tpl_size)*0.3
        >>> pprint(dummy_predictions_class_91)
        array([[[ 0.1,  0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1,  0.1]]])
        >>> pprint(dummy_predictions_class_109)
        array([[[ 0.2,  0.2,  0.2,  0.2],
                [ 0.2,  0.2,  0.2,  0.2],
                [ 0.2,  0.2,  0.2,  0.2],
                [ 0.2,  0.2,  0.2,  0.2],
                [ 0.2,  0.2,  0.2,  0.2]]])
        >>> pprint(dummy_predictions_class_150)
        array([[[ 0.3,  0.3,  0.3,  0.3],
                [ 0.3,  0.3,  0.3,  0.3],
                [ 0.3,  0.3,  0.3,  0.3],
                [ 0.3,  0.3,  0.3,  0.3],
                [ 0.3,  0.3,  0.3,  0.3]]])
        >>> 
        >>> p.get_labels()
        [91, 109, 150]
        >>> probmap_4d = np.array([dummy_predictions_class_91, dummy_predictions_class_109, dummy_predictions_class_150])
        >>> probmap_4d.shape #has to match 3 classes, 1 z slice, 5 x, 4 y 
        (3, 1, 5, 4)
        >>> p[0].put_probmap_data(probmap_4d) # dimensions are correct, put works
        '''

        if len(probmap_data.shape) != 4: 
            raise ValueError(\
                '''no valid dimension for probmap template: 
                   shape is %s, len of shape should be 4: (c,z,x,y)'''\
                                % str(probmap_data.shape))

        n_c, n_z, n_x, n_y = probmap_data.shape

        if n_c != len(self._labels):
            raise ValueError(\
                '''template must have %s channels, one channel for each
                   label in follwoing label order: %s'''\
                                % (str(len(self._labels)), str(self._labels)))

        if (n_z, n_x, n_y) != self._size_zxy:
            raise ValueError(\
                '''zxy shape of probmap template is not valid: 
                   is %s, should be %s''' \
                   % ((str((n_z, n_x, n_y)), str(self._size_zxy))))

        for data_layer, label in zip(probmap_data, self.get_labels()):
            worked = self.put_probmap_data_for_label(data_layer, label)            
            if not worked:
                logger.warning('could not write predictions for label %s', str(label))

    def put_probmap_data_for_label(self, probmap_data, label):
        if len(probmap_data.shape) != 3: 
            raise ValueError(\
                '''no valid dimension for probmap template: 
                   shape is %s, len of shape should be 3: (z,x,y)'''\
                                % str(probmap_data.shape))

        n_z, n_x, n_y = probmap_data.shape
        if (n_z, n_x, n_y) != self._size_zxy:
            raise ValueError(\
                '''zxy shape of probmap template is not valid: 
                   is %s, should be %s''' \
                   % ((str((n_z, n_x, n_y)), str(self._size_zxy))))    

            
        if label not in self._labels:
            raise ValueError('label %s not found in labels %s' % (str(label), str(self._labels)))

        return self._dataset.put_prediction_template(probmap_data, self._pos_zxy, self._image_nr, label)    
            
    def get_pixels(self):
        return self._dataset.get_multichannel_pixel_template(\
            self._image_nr, self._pos_zxy, self._size_zxy, self._channels,\
            pixel_padding=self._padding_zxy)   
        
    def _compute_pos_zxy(self):
        '''
        Compute all possible template positions for the whole dataset
        for template_size = self._size_zxy (size of output layer).
        
        :returns: list of template positions  

        e.g.
        [(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]

        '''
        tpl_pos = []
        for img_nr in list(range(self._dataset.n_images)):    
            img_shape_czxy = self._dataset.get_img_dimensions(img_nr)
            
            img_shape_zxy = img_shape_czxy[1:]
            
            tpl_pos =tpl_pos + [(img_nr, pos) for pos in ut.compute_pos(img_shape_zxy, self._size_zxy)]

        return tpl_pos    
    

    # def _is_tpl_size_valid(self, size_zxy):
    #     '''
    #     Checks if the size is not larger than the smallest image in the dataset
    #     '''
    #     for img_nr in list(range(self._dataset.n_images)):
            
    #         img_shape_czxy = self._dataset.get_img_dimensions(img_nr)
    #         img_shape_zxy = img_shape_czxy[1:]
    #         fits_in_image = (np.array(size_zxy) <= np.array(img_shape_zxy)).all()

    #         if not fits_in_image:
    #             return False
    #     return True        







