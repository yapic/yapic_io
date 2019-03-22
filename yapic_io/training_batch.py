import random
import numpy as np
from yapic_io.minibatch import Minibatch
from yapic_io.utils import compute_pos
import logging
import os

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)

class TrainingBatch(Minibatch):
    '''
    Infinite iterator providing pixel and label data for classifier training.

    * Provides tile data for classifier training.
    * Data is loaded from the dataset object.

    Parameters
    ----------
    dataset: Dataset
        Handle for reading pixels and labels weights.
    batch_size: int
        Nr of tiles for one batch.
    size_zxy: (nr_zslices, nr_x, nr_y)
        3d tile size (size of classifier output template).
    padding_zxy: (z, x, y)
        Growing of pixel tile in z, x and y (to define size of classifier input
        template relative to output template size size_zxy).
    equalized: bool
        If ``True``, less frequent labels are favored in randomized
        tile selection.

    Examples
    --------
    >>> from yapic_io import TiffConnector, Dataset, PredictionBatch
    >>> import tempfile
    >>>
    >>> # define data locations
    >>> pixel_img_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_img_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = tempfile.TemporaryDirectory()
    >>>
    >>> # size of network output layer in zxy
    >>> tile_size = (1, 5, 4) # size of network output layer in zxy
    >>>
    >>> # padding of network input layer in zxy, in respect to output layer
    >>> padding = (0, 2, 2)
    >>>
    >>> c = TiffConnector(pixel_img_dir, label_img_dir, savepath=savepath.name)
    >>> m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)
    >>> print(m)
    TrainingBatch (batch_size: 3, tile_size (zxy): (1, 5, 4), augment: {'flip'}
    >>>
    >>> for counter, mini in enumerate(m):
    ...     # shape of weights is (6, 3, 1, 5, 4) :
    ...     # batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
    ...     weights = mini.weights()
    ...
    ...     pixels = mini.pixels()
    ...     # shape of pixels is (6, 3, 1, 9, 8) :
    ...     # 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
    ...
    ...     # training on mini.pixels and mini.weights goes here
    ...     if counter > 10: #m is infinite
    ...         break
    '''

    def __init__(self,
                 dataset,
                 size_zxy,
                 padding_zxy=(0, 0, 0),
                 equalized=False):

        batch_size = len(dataset.label_values())

        super().__init__(dataset,
                         batch_size,
                         size_zxy,
                         padding_zxy=padding_zxy)

        self.equalized = equalized
        self.augmentation = set()
        self.augment_by_flipping(True)
        self.rotation_range = None
        self.shear_range = None
        self._pixels = None
        self._weights = None

        self.tile_pos_for_label = {key: self.tile_positions()
                                   for key in self.labels}

    def __repr__(self):
        info = ('TrainingBatch (batch_size: {}, '
                'tile_size (zxy): {}, augment: {}')
        return info.format(self._batch_size,
                           self.tile_size_zxy,
                           self.augmentation)

    def __iter__(self):
        return self

    def __next__(self):
        pixels = []
        weights = []
        augmentations = []

        for label in self.labels:
            tile_data = self._random_tile(for_label=label)

            pixels.append(tile_data.pixels)
            weights.append(tile_data.weights)
            augmentations.append(tile_data.augmentation)

        self._pixels = np.array(pixels)
        self._weights = np.array(weights, self.float_data_type)
        self.augmentations = augmentations

        return self

    def augment_by_flipping(self, flip_on):
        '''
        Data augmentation setting. Advantage: fast.

        Parameters
        ----------
        flip_on: bool
            If ``True``, tiles are randomly flipped for augmentation.
        '''
        if flip_on:
            self.augmentation.add('flip')
        else:
            self.augmentation.discard('flip')

    def augment_by_rotation(self, rot_on, rotation_range=(-45, 45)):
        '''
        Data augmentation setting. Slower than flipping, but you get more
        training data.

        Parameters
        ----------
        rot_on: bool
            If ``True``, tiles are randomly rotated.
        rotation_range: (min, max)
            Rotation angle range in degrees.
        '''
        self.rotation_range = rotation_range
        if rot_on:
            self.augmentation.add('rotate')
        else:
            self.augmentation.discard('rotate')

    def augment_by_shear(self, shear_on, shear_range=(-5, 5)):
        '''
        Data augmentation setting. Slower than flipping, but you get more
        training data.

        Parameters
        ----------
        shear_on: bool
            If ``True``, tiles are randomly sheared.
        shear_range: (min, max)
            Shear angle range in degrees.
        '''
        self.shear_range = shear_range
        if shear_on:
            self.augmentation.add('shear')
        else:
            self.augmentation.discard('shear')

    def pixels(self):
        pix = self._normalize(self._pixels).astype(self.float_data_type)

        return np.moveaxis(pix, [0, 1, 2, 3, 4],
                           self.pixel_dimension_order)


    def weights(self):
        return np.moveaxis(self._weights, [0, 1, 2, 3, 4],
                           self.pixel_dimension_order)

    def tile_positions(self, sliding=True):
        '''
        Get all possible sliding window tile positions of the dataset.

        Parameters
        ----------
        sliding: bool
            If True, sliding window positions are returned. Otherwise
            positions of non-overlapping tiles.

        Returns
        -------
        array_like
            List of 4-element tuples. Tuples define (image_id, z, x, y).
        '''

        im_sizes = np.zeros((self.dataset.n_images, 4), dtype=np.int16)
        tile_pos = []
        for i in range(self.dataset.n_images):
            img_shape = self.dataset.image_dimensions(i)[1:]
            pos = [(i, p[0], p[1], p[2])
                   for p in compute_pos(img_shape,
                                        self.tile_size_zxy,
                                        sliding_window=sliding)]
            tile_pos += pos

        return tile_pos

    def _augment_params(self):
        '''
        select random augmentation parameters
        '''

        augment_params = {}

        if 'flip' in self.augmentation:
            _, x, y = self.tile_size_zxy
            is_square_tile = (x == y)

            augment_params = {'fliplr': np.random.choice([True, False]),
                              'flipud': np.random.choice([True, False]),
                              'rot90': np.random.choice(4)
                              if is_square_tile else 0}

        if 'rotate' in self.augmentation:
            augment_params['rotation_angle'] = random.uniform(
                                                   *self.rotation_range)

        if 'shear' in self.augmentation:
            augment_params['shear_angle'] = random.uniform(*self.shear_range)

        return augment_params

    def _random_tile(self, for_label):
        '''
        Pick random tile in image regions where label data is present.
        '''

        # random pollng loop
        for counter in range(100):

            pos = self.tile_pos_for_label[for_label]
            choice = np.random.choice(range(len(pos)))
            pos_selected = pos[choice]
            image_nr = pos_selected[0]
            pos_zxy = pos_selected[1:]

            labels = np.array(sorted(self.labels))
            channels = np.array(sorted(self.channels))
            tile_data = self.dataset.training_tile(
                                    image_nr,
                                    pos_zxy,
                                    self.tile_size_zxy,
                                    channels,
                                    labels,
                                    pixel_padding=self.padding_zxy,
                                    augment_params=self._augment_params())

            # check if weights for specified label are present
            lblregion_index = np.where(tile_data.labels == for_label)
            weights_lblregion = tile_data.weights[lblregion_index]
            are_weights_in_tile = weights_lblregion.any()

            if are_weights_in_tile:
                msg = ('Needed {} trials to fetch random tile containing ' +
                       'labelvalue {}').format(counter,
                                               for_label)
                logger.info(msg)
                return tile_data

            else:
                # remove tile position for the label, since no labels here
                self.tile_pos_for_label[for_label].pop(choice)

        msg = ('Could not fetch random tile containing labelvalue {} ' +
               'within {} trials').format(for_label, counter)
        logger.warning(msg)
        return tile_data
