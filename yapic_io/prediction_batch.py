import yapic_io.utils as ut
import numpy as np
from numpy.testing import assert_equal
import os
import logging
from yapic_io.minibatch import Minibatch

logger = logging.getLogger(os.path.basename(__file__))


class PredictionBatch(Minibatch):
    '''
    List-like data interface for classification with neural networks.

    * Provides ``get_pixels()`` method for getting pixel tiles from
      4D images ``(channels, z, x, y)`` that fit into your neural network
      input layer.
    * Pixel data is loaded lazily to support arbitrary large image datasets.
    * Provides ``put_probmap_data()`` method to transfer classification
      results of your neural network back to the data source.
    * Flexible data binding through the Dataset object.

    Parameters
    ----------
    dataset: Dataset
        Handle for reading and writing pixels.
    batch_size: int
        Nr of tiles for one batch.
    size_zxy: (nr_zslices, nr_x, nr_y)
        3d tile size (size of classifier output template).
    padding_zxy: (z, x, y)
        Growing of pixel tile in z, x and y (to define size of classifier input
        template relative to output template size size_zxy).

    Examples
    --------
    >>> from yapic_io import TiffConnector, Dataset, PredictionBatch
    >>> import tempfile
    >>> # mock classification function
    >>> def classify(pixels, value):
    ...     return np.ones(pixels.shape) * value
    >>>
    >>>
    >>> # define data loacations
    >>> pixel_img_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_img_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = tempfile.TemporaryDirectory()
    >>>
    >>> # size of network output layer in zxy
    >>> tile_size = (1, 5, 4) # size of network output layer in zxy
    >>>
    >>> # padding of network input layer in zxy, in respect to output layer
    >>> padding = (0, 0, 0)
    >>>
    >>> # make training_batch mb and prediction interface p
    >>> # with TiffConnector binding
    >>> c = TiffConnector(pixel_img_dir, label_img_dir, savepath=savepath.name)
    >>> p = PredictionBatch(Dataset(c), 2, tile_size, padding_zxy=padding)
    >>> len(p)
    255
    >>> p.labels
    {1, 2, 3}
    >>>
    >>> # classify the whole bound dataset
    >>> counter = 0 # needed for mock data
    >>> for item in p:
    ...     pixels = item.pixels() # input for classifier
    ...     mock_classifier_result = classify(pixels, counter)
    ...     # pass classifier results for each class to data source
    ...     item.put_probmap_data(mock_classifier_result)
    ...     counter += 1

    See Also
    --------
    yapic.io.training_batch
    '''
    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0, 0, 0)):

        super().__init__(dataset,
                         batch_size,
                         size_zxy,
                         padding_zxy=padding_zxy)
        self.current_batch_pos = 0
        self.multichannel = False

        if size_zxy:
            self.set_tile_size(size_zxy)

    def multichannel_output_on(self):
        '''
        Probability maps are saved in multichannel images, one channel for each
        class.
        '''
        self.multichannel = True

    def multichannel_output_off(self):
        '''
        Probability maps are saved separate singlechannel images, one image for
        each class with suffix _class{class_nr}.tif.
        '''
        self.multichannel = False

    def set_tile_size(self, size_zxy):
        super().set_tile_size(size_zxy)
        self._all_tile_positions = self._compute_pos_zxy()

    def pixels(self):
        load_img = self.dataset.multichannel_pixel_tile

        pixels = [load_img(im_nr, pos_zxy,
                           self.tile_size_zxy,
                           self.channels,
                           self.padding_zxy)
                  for im_nr, pos_zxy in self.current_tile_positions]
        pixels = np.array(pixels)
        pixels = np.moveaxis(pixels, [0, 1, 2, 3, 4],
                             self.pixel_dimension_order)

        return self._normalize(pixels).astype(self.float_data_type)

    def __len__(self):
        '''
        Return the number of batches.
        '''
        n = len(self._all_tile_positions)  # nr of single tiles
        return int(np.ceil(n / self._batch_size))

    def __getitem__(self, position):
        '''
        Implements list-like operations of element selection and slicing.
        '''
        if position >= len(self):
            raise IndexError('index out of bounds')

        self.current_batch_pos = position
        return self

    @property
    def current_tile_positions(self):
        total = len(self._all_tile_positions)
        start = self.current_batch_pos * self._batch_size
        size = np.minimum(self._batch_size, total - start)

        return [self._all_tile_positions[x]
                for x in range(start, start + size)]

    def put_probmap_data(self, probmap_batch):
        '''
        Put classification results to the data source.



        Parameters
        ----------
        probmap_batch: ndarray
            5D matrix with shape (batches, nr_labels, z, x, y).

        Returns
        -------
        bool
            True if successful.

        Notes
        -----
        The order of the labels list (acessed with ``self.labels``) defines
        the order of the labels layer in the probability map.

        To pass 3D probmaps for a certain label, use
        ``put_probmap_data_for_label()``.
        '''

        # reorder dimensions to bczxy
        probmap_batch = np.moveaxis(probmap_batch,
                                    self.pixel_dimension_order,
                                    [0, 1, 2, 3, 4])
        nr_classes = probmap_batch.shape[1]
        if not self.multichannel:
            nr_classes = False

        assert_equal(len(probmap_batch.shape), 5, '5-dim (B,L,Z,X,Y) expected')
        B, L, *ZXY = probmap_batch.shape
        self.labels = np.arange(L) + 1 if len(self.labels) == 0 \
            else self.labels

        assert_equal(B, len(self.current_tile_positions))
        assert_equal(L, len(self.labels))
        assert_equal(ZXY, self.tile_size_zxy)

        for probmap, (image_nr, pos_zxy) in zip(probmap_batch,
                                                self.current_tile_positions):

            for label_ch, label in zip(probmap, self.labels):
                self.dataset.pixel_connector.put_tile(
                    label_ch,
                    pos_zxy,
                    image_nr,
                    label,
                    multichannel=nr_classes)

    def _compute_pos_zxy(self):
        '''
        Compute all possible tile positions for the whole dataset
        for tile_size = self._size_zxy (size of output layer).

        Returns
        -------
        list
            Tile positions
            e.g.
            ``[(image_nr, zpos, xpos, ypos),
               (image_nr, zpos, xpos, ypos), ...]``
        '''
        tile_pos = []
        for img_nr in list(range(self.dataset.n_images)):
            img_shape_zxy = self.dataset.image_dimensions(img_nr)[1:]
            tile_pos = tile_pos + [(img_nr, pos) for pos
                                   in ut.compute_pos(img_shape_zxy,
                                                     self.tile_size_zxy)]

        return tile_pos
