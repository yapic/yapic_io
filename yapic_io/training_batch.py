import random
import numpy as np
from yapic_io.minibatch import Minibatch




class TrainingBatch(Minibatch):
    '''
    Infinite iterator providing pixel and label data for classifier training.

    - Provides tile data for classifier training.

    - All data is loaded to memory on initialization of a training_batch.

    - Data is loaded from the dataset object.

    Code example for initializing a TrainingBatch:

    >>> from yapic_io.factories import make_tiff_interface
    >>>
    >>> # define data locations
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = 'yapic_io/test_data/tmp/'
    >>>
    >>> tile_size = (1, 5, 4) # size of network output layer in zxy
    >>> padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer
    >>>
    >>> # make training_batch mb and prediction interface p with TiffConnector binding
    >>> m, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tile_size, padding_zxy=padding)
    >>>
    >>> counter=0
    >>> for mini in m:
    ...     weights = mini.weights() #shape is (6, 3, 1, 5, 4) : 3 label-classes, 1 z, 5 x, 4 y
    ...     #shape of weights is (6, 3, 1, 5, 4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
    ...
    ...     pixels = mini.pixels()
    ...     # shape of pixels is (6, 3, 1, 9, 8) : 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
    ...     # here: apply training on mini.pixels and mini.weights
    ...     counter += 1
    ...     if counter > 10: #m is infinite
    ...         break
    '''

    def __init__(self,
                 dataset,
                 size_zxy,
                 padding_zxy=(0, 0, 0),
                 augment=True,
                 augment_simple=True,
                 rotation_range=(-45, 45),
                 shear_range=(-5, 5),
                 equalized=False):
        '''
        :param dataset: dataset object, to connect to pixels and labels weights
        :type dataset: Dataset
        :param batch_size: nr of tiles
        :type batch_size: int
        :param size_zxy: 3d tile size (size of classifier output tmeplate)
        :type size_zxy: tuple (with length 3)
        :param padding_zxy: growing of pixel tile in (z, x, y).
        :type padding_zxy: tuple (with length 3)
        :param augment: if True, tiles are randomly rotatted and sheared
        :type augment: bool
        :param augment_simple: if True, tiles are randomly flipped (faster than augment)
        :type augment_simple: bool
        :param rotation_range: range of random rotation in degrees (min_angle, max_angle)
        :type rotation_angle: tuple (with length 2)
        :param shear_range: range of random shear in degrees (min_angle, max_angle)
        :type shear_angle: tuple (with length 2)
        :param equalized: if True, less frequent labels are favored in randomized tile selection
        :type equalized: bool
        '''
        batch_size = len(dataset.label_values())

        super().__init__(dataset, batch_size, size_zxy, padding_zxy=padding_zxy)

        self.equalized = equalized
        self.augment = augment
        self.augment_simple = augment_simple
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self._pixels = None
        self._weights = None

        #self._fetch_training_batch_data()


    def __repr__(self):
        infostring = ('TrainingBatch\n'
                      '  batch_size: {}\n'
                      '  tile size (size_zxy): {}\n'
                      '  augment: {}\n').format(
             self._batch_size, self._size_zxy, self.augment)
        return infostring


    def __iter__(self):
        return self


    def __next__(self):
        self._fetch_training_batch_data()
        return self


    def pixels(self):
        return self._normalize(self._pixels.astype(self.float_data_type))


    def weights(self):
        return self._weights.astype(self.float_data_type)


    def _fetch_training_batch_data(self):
        pixels = []
        weights = []
        augmentations = []

        for label in self._labels:
            tile_data = self._random_tile(for_label=label)
            pixels.append(tile_data.pixels)
            weights.append(tile_data.weights)
            augmentations.append(tile_data.augmentation)

        self._pixels = np.array(pixels)
        self._weights = np.array(weights)
        self.augmentations = augmentations


    def _random_rotation_angle(self):
        '''
        get random rotation angle within specified range
        '''
        if not self.augment:
            return 0

        return random.uniform(*self.rotation_range)

    def _random_shear_angle(self):
        '''
        get random shear angle within specified range
        '''
        if not self.augment:
            return 0

        return random.uniform(*self.shear_range)


    def _random_simple_augment_params(self):
        '''
        get random parameters for simple augmentation
        '''
        fliplr = False
        flipud = False
        rot90 = 0

        if not self.augment_simple:
            return fliplr, flipud, rot90


        rot90 = np.random.choice(4)

        if np.random.choice(2) == 0:
            fliplr = True
        if np.random.choice(2) == 0:
            flipud = True

        return fliplr, flipud, rot90





    def _random_tile(self, for_label=None):
        '''
        pick random tile in image regions where label data is present
        '''
        if not self.augment:
            return self._dataset.random_training_tile(self._size_zxy,
                                                          self._channels,
                                                          pixel_padding=self._padding_zxy,
                                                          equalized=self.equalized,
                                                          labels=self._labels,
                                                          label_region=for_label)

        shear_angle = self._random_shear_angle()
        rotation_angle = self._random_rotation_angle()
        fliplr, flipud, rot90 = self._random_simple_augment_params()

        augment_params = {'shear_angle' : shear_angle,
                          'rotation_angle' : rotation_angle,
                          'fliplr' : fliplr,
                          'flipud' : flipud,
                          'rot90' : rot90}

        return self._dataset.random_training_tile(self._size_zxy,
                     self._channels,
                     pixel_padding=self._padding_zxy,
                     equalized=self.equalized,
                     augment_params=augment_params,
                     labels=self._labels,
                     label_region=for_label)

