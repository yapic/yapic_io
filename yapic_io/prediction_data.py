import yapic_io.utils as ut
import numpy as np

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
        self._padding_zxy = padding_zxy
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

    def put_probmap_data(self, probmap_data):
        '''
        Put classification results to the data source.

        :param probmap_data: 4D matrix with shape (nr_labels, z,x,y).
        :type probmap_data: numpy.array.
        :returns: bool (True if successful).
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
    

    def _is_tpl_size_valid(self, size_zxy):
        '''
        Checks if the size is not larger than the smallest image in the dataset
        '''
        for img_nr in list(range(self._dataset.n_images)):
            
            img_shape_czxy = self._dataset.get_img_dimensions(img_nr)
            img_shape_zxy = img_shape_czxy[1:]
            fits_in_image = (np.array(size_zxy) <= np.array(img_shape_zxy)).all()

            if not fits_in_image:
                return False
        return True        







