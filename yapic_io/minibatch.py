import random

class Minibatch(object):
    '''
    
    Provides template data for classifier training.
    
    All data is loaded to memory on initialization of a minibatch.

    Data is loaded from the dataset object.




    
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

        
        self.batch_size = batch_size
        self.equalized = equalized

        self.augment = augment
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        
        self._template_data = [self._pick_random_tpl() for _ in list(range(self.batch_size))]
        
        

    def __repr__(self):

        infostring = \
            'Minibatch \n' \
            'batch_size: %s \n' \
            'template size (size_zxy): %s \n' \
            'augment: %s \n' \
             % (self.batch_size, self.size_zxy, self.augment)\
               

        return infostring


    def __len__(self):
        return self.batch_size

    
    def __getitem__(self, position):
        return self._template_data[position]
 


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
                 equalized=self.equalized)       

        shear_angle = self._get_random_shear()
        rotation_angle = self._get_random_rotation()    
        
        return self.dataset.pick_random_training_template(self.size_zxy\
            , self.channels, pixel_padding=self.padding_zxy\
            , equalized=self.equalized, rotation_angle=rotation_angle\
            , shear_angle=shear_angle)       
        
                
