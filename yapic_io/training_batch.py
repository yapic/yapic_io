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

    >>> from yapic_io import TiffConnector, Dataset, PredictionBatch
    >>> import tempfile
    >>>
    >>> # define data locations
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = tempfile.TemporaryDirectory()
    >>>
    >>> tile_size = (1, 5, 4) # size of network output layer in zxy
    >>> padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer
    >>>
    >>> c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath.name)
    >>> m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)
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
        :type shear_angle: tuple (with length 2)
        :param equalized: if True, less frequent labels are favored in randomized tile selection
        :type equalized: bool
        '''
        batch_size = len(dataset.label_values())

        super().__init__(dataset, batch_size, size_zxy, padding_zxy=padding_zxy)

        self.equalized = equalized
        self.augmentation = set()
        self.augment_by_flipping(True)
        self.rotation_range = None
        self.shear_range = None
        self._pixels = None
        self._weights = None

        #self._fetch_training_batch_data()


    def __repr__(self):
        info = 'TrainingBatch (batch_size: {}, tile_size (zxy): {}, augment: {}'
        return info.format(self._batch_size, self.tile_size_zxy, self.augmentation)


    def __iter__(self):
        return self


    def __next__(self):
        self._fetch_training_batch_data()
        return self

    def augment_by_flipping(self, flip_on):
        '''
        :param flip_on: if True, tiles are randomly flipped (fast)
        '''
        if flip_on:
            self.augmentation.add('flip')
        else:
            self.augmentation.discard('flip')

    def augment_by_rotation(self, rot_on, rotation_range=(-45,45)):
        '''
        :param rot_on: if True, tiles are randomly rotated
        :param rotation_range: (min, max) rotation angle in degrees
        '''
        self.rotation_range = rotation_range
        if rot_on:
            self.augmentation.add('rotate')
        else:
            self.augmentation.discard('rotate')

    def augment_by_shear(self, shear_on, shear_range=(-5, 5)):
        '''
        :param shear_on: if True, tiles are randomly sheared
        :param shear_range: (min, max) shear angle in degrees
        '''
        self.shear_range = shear_range
        if shear_on:
            self.augmentation.add('shear')
        else:
            self.augmentation.discard('shear')


    def pixels(self):
        return self._normalize(self._pixels).astype(self.float_data_type)


    def weights(self):
        return self._weights.astype(self.float_data_type)


    def _fetch_training_batch_data(self):
        pixels = []
        weights = []
        augmentations = []

        for label in self.labels:
            tile_data = self._random_tile(for_label=label)

            pixels.append(tile_data.pixels)
            weights.append(tile_data.weights)
            augmentations.append(tile_data.augmentation)

        self._pixels = np.array(pixels)
        self._weights = np.array(weights)
        self.augmentations = augmentations


    def _random_tile(self, for_label=None):
        '''
        pick random tile in image regions where label data is present
        '''
        augment_params = {}

        if 'flip' in self.augmentation:
            _, x, y = self.tile_size_zxy
            is_square_tile = (x == y)

            augment_params = {'fliplr' : np.random.choice([True, False]),
                              'flipud' : np.random.choice([True, False]),
                              'rot90' : np.random.choice(4) if is_square_tile else 0}

        if 'rotate' in self.augmentation:
            augment_params['rotation_angle'] = random.uniform(*self.rotation_range)

        if 'shear' in self.augmentation:
             augment_params['shear_angle'] = random.uniform(*self.shear_range)

        return self._dataset.random_training_tile(self.tile_size_zxy,
                                                  self.channels,
                                                  pixel_padding=self.padding_zxy,
                                                  equalized=self.equalized,
                                                  augment_params=augment_params,
                                                  labels=self.labels,
                                                  label_region=for_label)
