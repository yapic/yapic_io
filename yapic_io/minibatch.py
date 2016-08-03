import random
import numpy as np
class Minibatch(object):
    '''
    Infinite iterator providing pixel and label data for classifier training.

    - Provides template data for classifier training.

    - All data is loaded to memory on initialization of a minibatch.
    
    - Data is loaded from the dataset object.

    Code example for initializing a Minibatch:

    >>> from yapic_io.tiff_connector import TiffConnector
    >>> from yapic_io.dataset import Dataset
    >>> 
    >>> pixel_image_dir = '/path/to/my/tiff/images/'
    >>> label_image_dir = '/path/to/my/label/images/'
    >>> t = TiffConnector(pixel_image_dir, label_image_dir)
    >>> d = Dataset(t)
    >>>
    >>> size = (1,3,4)
    >>> pad = (1,2,2)
    >>> batch_size = 4
    >>>
    >>> m = Minibatch(d, batch_size, size, padding_zxy=pad)
    >>> 
    >>> c=0
    >>> for mini in m:
    >>>     print(mini.pixels)
    >>>     print(mini.weights)
    >>>     c+=1
    >>>     if c>10:
    >>>         break





    
    '''
    

    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0,0,0),
        augment=True, rotation_range=(-45,45), shear_range=(-5,5), equalized=False):
        
        '''
        :param dataset: dataset object, to connect to pixels and labels weights 
        :type dataset: Dataset
        :param batch_size: nr of templates
        :type batch_size: int
        :param size_zxy: 3d template size (size of classifier output tmeplate)
        :type size_zxy: tuple (with length 3)
        :param padding_zxy: growing of pixel template in (z,x,y).
        :type padding_zxy: tuple (with length 3)
        :param augment: if True, templates are randomly rotatted and sheared
        :type augment: bool
        :param rotation_range: range of random rotation in degrees (min_angle,max_angle) 
        :type rotation_angle: tuple (with length 2)
        :param shear_range: range of random shear in degrees (min_angle,max_angle) 
        :type shear_angle: tuple (with length 2)
        :param equalized: if True, less frequent labels are favored in randomized template selection
        :type equalized: bool
        '''
        self.dataset = dataset

        self.padding_zxy = padding_zxy
        self.size_zxy = size_zxy
        self.channels = self.dataset.get_channels()
        self.labels = self.dataset.get_label_values()
        
        self.batch_size = batch_size
        self.equalized = equalized

        self.augment = augment
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        
        self.pixels = None
        self.weights = None

        self._fetch_minibatch_data() 
        #self._template_data = [self._pick_random_tpl() for _ in list(range(self.batch_size))]
        
        

    def __repr__(self):

        infostring = \
            'Minibatch \n' \
            'batch_size: %s \n' \
            'template size (size_zxy): %s \n' \
            'augment: %s \n' \
             % (self.batch_size, self.size_zxy, self.augment)\
               

        return infostring

    def __iter__(self):
        return self

    def __next__(self):
        self._fetch_minibatch_data()
        return self
        
 
    def _fetch_minibatch_data(self):
        pixels = []
        weights = []
        augmentations = []
        for i in list(range(self.batch_size)):
            tpl_data = self._pick_random_tpl()
            pixels.append(tpl_data.pixels)
            weights.append(tpl_data.weights)
            augmentations.append(tpl_data.augmentation)

        self.pixels = np.array(pixels)
        self.weights = np.array(weights)    
        self.augmentations = augmentations    



    def _get_random_rotation(self):
        '''
        get random rotation angle within specified range
        '''
        return random.uniform(*self.rotation_range)

    def _get_random_shear(self):
        '''
        get random shear angle within specified range
        '''
        return random.uniform(*self.shear_range)  



    def _pick_random_tpl(self):
        '''
        pick random template in image regions where label data is present

        '''
        
        if not self.augment:
            return self.dataset.pick_random_training_template(self.size_zxy\
            , self.channels, pixel_padding=self.padding_zxy,\
                 equalized=self.equalized, labels=self.labels)       

        shear_angle = self._get_random_shear()
        rotation_angle = self._get_random_rotation()    
        
        return self.dataset.pick_random_training_template(self.size_zxy\
            , self.channels, pixel_padding=self.padding_zxy\
            , equalized=self.equalized, rotation_angle=rotation_angle\
            , shear_angle=shear_angle, labels=self.labels)       
        
                
