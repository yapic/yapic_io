import random
import numpy as np
from yapic_io.minibatch import Minibatch


class TrainingBatch(Minibatch):
    '''
    Infinite iterator providing pixel and label data for classifier training.

    - Provides template data for classifier training.

    - All data is loaded to memory on initialization of a training_batch.

    - Data is loaded from the dataset object.

    Code example for initializing a TrainingBatch:

    >>> from yapic_io.factories import make_tiff_interface
    >>>
    >>> #define data locations
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = 'yapic_io/test_data/tmp/'
    >>>
    >>> tpl_size = (1, 5, 4) # size of network output layer in zxy
    >>> padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer
    >>>
    >>> # make training_batch mb and prediction interface p with TiffConnector binding
    >>> m, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size, padding_zxy=padding)
    >>>
    >>> counter=0
    >>> for mini in m:
    ...     weights = mini.weights() #shape is (6, 3, 1, 5, 4) : 3 label-classes, 1 z, 5 x, 4 y
    ...     #shape of weights is (6, 3, 1, 5, 4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
    ...
    ...     pixels = mini.pixels()
    ...     # shape of pixels is (6, 3, 1, 9, 8) : 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
    ...     #here: apply training on mini.pixels and mini.weights
    ...     counter += 1
    ...     if counter > 10: #m is infinite
    ...         break






    '''


    def __init__(self, dataset, size_zxy, padding_zxy=(0, 0, 0), 
        augment=True, rotation_range=(-45, 45), shear_range=(-5, 5), equalized=False, \
        ):

        '''
        :param dataset: dataset object, to connect to pixels and labels weights
        :type dataset: Dataset
        :param batch_size: nr of templates
        :type batch_size: int
        :param size_zxy: 3d template size (size of classifier output tmeplate)
        :type size_zxy: tuple (with length 3)
        :param padding_zxy: growing of pixel template in (z, x, y).
        :type padding_zxy: tuple (with length 3)
        :param augment: if True, templates are randomly rotatted and sheared
        :type augment: bool
        :param rotation_range: range of random rotation in degrees (min_angle, max_angle)
        :type rotation_angle: tuple (with length 2)
        :param shear_range: range of random shear in degrees (min_angle, max_angle)
        :type shear_angle: tuple (with length 2)
        :param equalized: if True, less frequent labels are favored in randomized template selection
        :type equalized: bool
        '''

        batch_size = len(dataset.label_values())

        super().__init__(dataset, batch_size, size_zxy, padding_zxy=padding_zxy)

        self.equalized = equalized

        self.augment = augment
        self.rotation_range = rotation_range
        self.shear_range = shear_range

        self._pixels = None
        self._weights = None

        self._fetch_training_batch_data()
        #self._template_data = [self._pick_random_tpl() for _ in list(range(self._batch_size))]



    def __repr__(self):

        infostring = \
            'TrainingBatch \n' \
            'batch_size: %s \n' \
            'template size (size_zxy): %s \n' \
            'augment: %s \n' \
             % (self._batch_size, self._size_zxy, self.augment)\


        return infostring

    def __iter__(self):
        return self

    def __next__(self):
        self._fetch_training_batch_data()
        return self


    def pixels(self):
        return self._pixels.astype(self.float_data_type)

    def weights(self):
        return self._weights.astype(self.float_data_type)



    def _fetch_training_batch_data(self):
##        print('-------------------- _fetch_training_batch_data --------------------')
        pixels = []
        weights = []
        augmentations = []
        # for i in list(range(self._batch_size)):
        #     tpl_data = self._pick_random_tpl()
        #     pixels.append(tpl_data.pixels)
        #     weights.append(tpl_data.weights)
        #     augmentations.append(tpl_data.augmentation)
        for label in self._labels:
##        for label in self._labels):
#        for label in [np.random.choice(self._labels)]:
##            print('-------------------- _fetch_training_batch_data: {} --------------------'.format(label))
            tpl_data = self._pick_random_tpl(for_label=label)
            pixels.append(tpl_data.pixels)
            weights.append(tpl_data.weights)
            augmentations.append(tpl_data.augmentation)

        self._pixels = np.array(pixels)
        self._weights = np.array(weights)
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



    def _pick_random_tpl(self, for_label=None):
        '''
        pick random template in image regions where label data is present

        '''

        if not self.augment:
            return self._dataset.random_training_template(self._size_zxy\
            , self._channels, pixel_padding=self._padding_zxy, \
                 equalized=self.equalized, labels=self._labels, label_region=for_label)

        shear_angle = self._get_random_shear()
        rotation_angle = self._get_random_rotation()

        return self._dataset.random_training_template(self._size_zxy\
            , self._channels, pixel_padding=self._padding_zxy\
            , equalized=self.equalized, rotation_angle=rotation_angle\
            , shear_angle=shear_angle, labels=self._labels, label_region=for_label)
