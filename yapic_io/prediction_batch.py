import yapic_io.utils as ut
import numpy as np
from numpy.testing import assert_equal, assert_array_less
from bisect import insort_left
import os
import logging
from yapic_io.minibatch import Minibatch

logger = logging.getLogger(os.path.basename(__file__))

class PredictionBatch(Minibatch):
    '''
    List-like data interface for classification with neural networks.

    - Provides get_pixels() method for getting pixel tiles from
      4D images (channels, z, x, y) that fit into your neural network
      input layer.

    - Pixel data is loaded lazily to support arbitrary large image datasets.

    - Provides put_probmap_data() method to transfer classification results of your
      neural network back to the data source.

    - Flexible data binding through the Dataset object.


        >>> from yapic_io.factories import make_tiff_interface
        >>> import tempfile
        >>> # mock classification function
        >>> def classify(pixels, value):
        ...     return np.ones(pixels.shape) * value
        >>>
        >>>
        >>> # define data loacations
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
        >>> savepath = tempfile.TemporaryDirectory()
        >>>
        >>> tile_size = (1, 5, 4) # size of network output layer in zxy
        >>> padding = (0, 0, 0) # padding of network input layer in zxy, in respect to output layer
        >>>
        >>> # make training_batch mb and prediction interface p with TiffConnector binding
        >>> _, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath.name, tile_size, padding_zxy=padding, training_batch_size=2)
        >>> len(p)
        255
        >>> p.labels
        [1, 2, 3]
        >>> # classify the whole bound dataset
        >>> counter = 0 # needed for mock data
        >>> for item in p:
        ...     pixels = item.pixels() # input for classifier
        ...     mock_classifier_result = classify(pixels, counter) # classifier output
        ...     # pass classifier results for each class to data source
        ...     item.put_probmap_data(mock_classifier_result)
        ...     counter += 1
    '''
    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0, 0, 0)):
        '''
        :param dataset: Dataset object for data binding of
                        image pixel data and classification results.
        :type dataset: Dataset
        :param size_zxy: zxy_size of the neural network output layer.
        :type size_zxy: tuple of 3 integers (z, x, y)
        :param padding_zxy: The padding of the network's input layer
                            relative to its output layer.
        :type padding_zxy: Tuple of 3 integers (z, x, y).
        '''
        super().__init__(dataset, batch_size, size_zxy, padding_zxy=padding_zxy)

        self.curr_batch_pos = 0 # current bach position


    def pixels(self):
        pixfunc = self._dataset.multichannel_pixel_tile

        pixels = [pixfunc(im_nr,
                          pos_zxy,
                          self._size_zxy,
                          self.channel_list,
                          self._padding_zxy)
                  for im_nr, pos_zxy in self._get_curr_tile_positions()]

        pixels = np.array(pixels)

        return self._normalize(pixels).astype(self.float_data_type)


    def __len__(self):
        '''
        Return the number of batches.
        '''
        n = len(self._all_tile_positions) # nr of single tiles
        return int(np.ceil(float(n) / self._batch_size))


    def __getitem__(self, position):
        '''
        Implements list-like operations of element selection and slicing.
        '''
        if position >= len(self):
            raise StopIteration('index out of bounds')

        self.curr_batch_pos = position

        return self


    def set_tile_size_zxy(self, size_zxy):
        '''
        overloads the set method for the _size_zxy attribute
        by updating the tile position list
        '''
        super().set_tile_size_zxy(size_zxy)

        # a list of all possible tile positions
        # [(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        self._all_tile_positions = self._compute_pos_zxy()


    def get_actual_batch_size(self):
        '''
        Returns the batch size (which might be smaller than `batch_size`
        if we only have `n < batch_size` tiles left)
        '''
        total = len(self._all_tile_positions) # nr of single tiles
        processed = self.curr_batch_pos * self._batch_size
        return np.minimum(self._batch_size, total - processed)


    def _get_curr_tile_positions(self):
        return [self._all_tile_positions[x] for x in self._get_curr_tile_indices()]


    def _get_curr_tile_indices(self):
        size = self.get_actual_batch_size()
        start = self.curr_batch_pos * self._batch_size
        return np.arange(start, start + size).tolist()


    def put_probmap_data(self, probmap_data):
        '''
        Put classification results to the data source.

        The order of the labels list (accesed with self.labels)defines
        the order of the labels layer in the probability map.

        To pass 3D probmaps for a certain label, use put_probmap_data_for_label()

        :param probmap_data: 4D matrix with shape (nr_labels, z, x, y).
        :type probmap_data: numpy.array.
        :returns: bool (True if successful).
        '''
        assert_equal(len(probmap_data.shape), 5, 'no valid dimension for probmap tile: 5-dim (B,L,Z,X,Y) expected')

        B, L, *ZXY= probmap_data.shape

        if len(self.labels) == 0:
            self.labels = np.arange(L)+1

        assert_equal(B, self.get_actual_batch_size(), 'Invalid batch size given')
        assert_equal(L, len(self.labels), 'Invalid number of labels given')
        assert_equal(ZXY, self._size_zxy, 'zxy shape of probmap tile is not valid')

        # iterate through batch
        for probmap_data_sel, tile_pos_index in zip(probmap_data, self._get_curr_tile_indices()):
            # iterate through label channels
            for data_layer, label in zip(probmap_data_sel, self.labels):
                self._put_probmap_data_for_label(data_layer, label, tile_pos_index)


    def _put_probmap_data_for_label(self, probmap_data, label, tile_pos_index):
        assert_array_less(tile_pos_index, len(self._all_tile_positions))

        assert_equal(len(probmap_data.shape), 3, 'no valid dimension for probmap tile: 3-dim (Z,X,Y) expected')
        assert_equal(probmap_data.shape, self._size_zxy, 'zxy shape for probmap tile invalid')

        if label not in self.labels:
            raise ValueError('label {} not found in labels {}'.format(label, self.labels))

        image_nr, pos_zxy = self._all_tile_positions[tile_pos_index]

        return self._dataset.put_prediction_tile(probmap_data, pos_zxy, image_nr, label)


    def _compute_pos_zxy(self):
        '''
        Compute all possible tile positions for the whole dataset
        for tile_size = self._size_zxy (size of output layer).

        :returns: list of tile positions

        e.g.
        [(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        '''
        tile_pos = []
        for img_nr in list(range(self._dataset.n_images)):
            img_shape_czxy = self._dataset.image_dimensions(img_nr)
            img_shape_zxy = img_shape_czxy[1:]

            tile_pos = tile_pos + [(img_nr, pos) for pos in ut.compute_pos(img_shape_zxy, self._size_zxy)]

        return tile_pos


