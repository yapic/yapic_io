import yapic_io.utils as ut
import numpy as np
from bisect import insort_left
import os
import logging
from yapic_io.minibatch import Minibatch

logger = logging.getLogger(os.path.basename(__file__))

class PredictionBatch(Minibatch):
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
        >>> # make training_batch mb and prediction interface p with TiffConnector binding
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
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=1)
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=2)
        ...     item.put_probmap_data_for_label(mock_classifier_result, label=3)
        ...     counter += 1
        >>>

    
    '''
    

    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0,0,0)):

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

        super().__init__(dataset, batch_size, size_zxy, padding_zxy=padding_zxy)
        
        self._pos_zxy = None 
        self._batch_nr = None 

        #a list of all possible template positions 
        #[(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        self._tpl_pos_all = self._compute_pos_zxy()
        
        #indices for template positions, organized in batches of defined size
        self._batch_index_list = self._get_batch_index_list() 


        



    def _get_batch_index_list(self):
        

        n = len(self._tpl_pos_all) #nr of single templates

        return ut.nest_list(list(range(n)), self._batch_size)





        
        
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
       

    
    def set_tpl_size_zxy(self, size_zxy):
        '''
        overloads the set method for the _size_zxy attribute
        by updating the template position list

        '''
        
        super().set_tpl_size_zxy(size_zxy)

        #a list of all possible template positions 
        #[(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        self._tpl_pos_all = self._compute_pos_zxy()   



        
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
        >>> # make training_batch mb and prediction interface p
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size)
        >>> #upon object initialization all available classes are set
        >>> p.get_labels()
        [1, 2, 3]
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
        [1, 2, 3]
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
    

    







