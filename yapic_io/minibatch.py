from bisect import insort_left
import os
import logging
import numpy as np
logger = logging.getLogger(os.path.basename(__file__))


class Minibatch(object):

    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0, 0, 0)):
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
        :param rotation_range: range of random rotation in degrees (min_angle, max_angle)
        :type rotation_angle: tuple (with length 2)
        :param shear_range: range of random shear in degrees (min_angle, max_angle)
        :type shear_angle: tuple (with length 2)
        :param equalized: if True, less frequent labels are favored in randomized tile selection
        :type equalized: bool
        '''
        self._dataset = dataset
        self._batch_size = batch_size
        self.normalize_mode = None
        self.global_norm_minmax = None

        self.float_data_type = np.float32  # type of pixel and weight data
        self._size_zxy = None
        self._padding_zxy = None
        if size_zxy is not None:
            self.set_tile_size_zxy(size_zxy)
        if padding_zxy is not None:
            self.set_padding_zxy(padding_zxy)
        # imports all available channels by default
        self._channels = self._dataset.channel_list()
        # imports all available labels by default
        self._labels = self._dataset.label_values()

    def set_normalize_mode(self, mode_str, minmax=None):
        '''
        local : scale between 0 and 1 (per minibatch, each channel
                separately)
        local_z_score : mean to zero, scale by standard deviation
                        (per minibatch, each channel separately)
        global : scale minmax=(min, max) between 0 and 1
                 whole dataset, each channel separately
        minmax : tuple of minimum and maximum pixel value for global
                 normalization. e.g (0, 255) for 8 bit images
        '''

        if mode_str in ('local_z_score', 'local', 'off'):
            self.normalize_mode = mode_str

        elif mode_str == 'global':
            if minmax is None:
                err_msg = 'Missing minmax for defining global minimum' + \
                          'and maximum for normalization: (min, max)'
                raise ValueError(err_msg)
            self.global_norm_minmax = minmax
            self.normalize_mode = mode_str

        else:
            err_msg = '''Wrong normalization mode argument: {}. '''.format(mode_str) + \
                      '''choose between 'local', 'local_z_score', 'global' and 'off' '''
            raise ValueError(err_msg)

    def get_tile_size_zxy(self):
        '''
        Returns the size in (z, x, y) of the tile. Should match the size in (z, x, y)
        of the network's output layer.
        '''
        return self._size_zxy

    def set_tile_size_zxy(self, size_zxy):
        '''
        Sets the size in (z, x, y) of the tile. Should match the size in (z, x, y)
        of the network's output layer.

        Must not be larger than the smallest image of the dataset!!
        '''
        if len(size_zxy) != 3:
            raise ValueError(
                '''no valid size for probmap tile:
                   shape is %s, len of shape should be 3: (z, x, y)'''
                % str(size_zxy))

        self._size_zxy = size_zxy

        # a list of all possible tile positions
        #[(image_nr, zpos, xpos, ypos), (image_nr, zpos, xpos, ypos), ...]
        #self._all_tile_positions = self._compute_pos_zxy()

    def get_padding_zxy(self):
        '''
        Returns the padding size in (z, x, y), i.e. the padding of the network's
        input layer compared to its output layer.

        Padding size has to be defined by the user according to the used network
        structure.

        '''
        return self._padding_zxy

    def set_padding_zxy(self, padding_zxy):
        '''
        Sets the padding size in (z, x, y), i.e. the padding of the network's
        input layer compared to its output layer.

        Padding size has to be defined according to the used network
        structure.

        Example: If the output layer's size is (1, 6, 4) and the input layer's
                 size is (1, 8, 6), then the padding in xyz is (0, 1, 1), i.e
                 the input layer grows 0 pixels in z, and 1 pixel in
                 in x and y, compared to the output layer. (The growth of
                 1 pixel at both ends results in a matrix that is 2 pixels larger
                 than the output layer).

                >>> import numpy as np
                >>> from pprint import pprint
                >>>
                >>> output_layer = np.zeros((1, 6, 4))
                >>> input_layer = np.zeros((1, 8, 6)) #padded by (0, 1, 1) compared to output_layer
                >>>
                >>> output_layer_expected = np.array([[[ 0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.]]])
                >>> input_layer_expected = np.array([[[ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.],
                ...     [ 0.,  0.,  0.,  0.,  0.,  0.]]])
                >>> np.testing.assert_array_equal(output_layer, output_layer_expected)
                >>> np.testing.assert_array_equal(input_layer, input_layer_expected)

        '''
        if len(padding_zxy) != 3:
            raise ValueError(
                '''no valid dimension for padding:
                   shape is %s, len of shape should be 3: (z, x, y)'''
                % str(padding_zxy.shape))
        self._padding_zxy = padding_zxy

    def channel_list(self):
        '''
        Returns the selected image channels.
        Per default, all available channels are selected.

        The order of the channels list defines the order of the
        channels layer in the pixels (accessed with get_pixels()).
        Pixels have the dimensionality (channels, z, x, y)

        >>> from yapic_io.factories import make_tiff_interface
        >>> import tempfile
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = tempfile.TemporaryDirectory()
        >>>
        >>> tile_size = (1, 5, 4)
        >>> # make training_batch mb and prediction interface p: upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath.name, tile_size) #upon object initialization all available image channels are set
        >>>
        >>>
        >>> p.channel_list() #we have 3 channels in the prediction interface
        [0, 1, 2]
        >>>
        >>> p[0].pixels().shape #accordingly, we have 3 channels in the 5D pixels tile (n, c, z, x, y)
        (10, 3, 1, 5, 4)
        >>>
        '''
        return self._channels

    def remove_channel(self, channel):
        '''
        Removes a pixel channel from the selection.
        :param channel: channel
        :returns: bool, True if channel was removed from selection

        >>> from yapic_io.factories import make_tiff_interface
        >>> import tempfile
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = tempfile.TemporaryDirectory()
        >>>
        >>> tile_size = (1, 5, 4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath.name, tile_size)
        >>>
        >>>
        >>> p.channel_list() #we have 3 channels
        [0, 1, 2]
        >>> p[0].pixels().shape #accordingly, we have 3 channels in the 5D pixels tile (n, c, z, x, y)
        (10, 3, 1, 5, 4)
        >>>
        >>> p.remove_channel(1) #remove channel 1
        True
        >>> p.channel_list() #only 2 channels selected
        [0, 2]
        >>> p[0].pixels().shape #only 2 channels left in the pixel tile
        (10, 2, 1, 5, 4)

        '''
        if channel not in self._channels:
            raise ValueError('not possible to remove channel %s from channel selection %s'
                             % (str(channel), str(self._channels)))
        self._channels.remove(channel)
        return True

    def add_channel(self, channel):
        '''
        Adds a pixel-channel to the selection.


        '''
        if channel in self._channels:
            logger.warning('channel already selected %s', channel)
            return False
        if channel not in self._dataset.channel_list():
            raise ValueError('not possible to add channel %s from dataset channels %s'
                             % (str(channel), str(self._dataset.channel_list())))

        insort_left(self._channels, channel)
        return True

    def get_labels(self):
        '''
        Returns the selected labels (i.e. classes for prediction).
        Per default, all available labels are selected.

        The order of the labels list defines the order of the
        labels layer in the probability map (i.e. the classification result
        matrix) that can be exported with put_probmap_data().
        '''

        return self._labels

    def remove_label(self, label):
        '''
        Removes a label class from the selection.

        :param label: label
        :returns: bool, True if label was removed from selection

        >>> from yapic_io.factories import make_tiff_interface
        >>> import tempfile
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = tempfile.TemporaryDirectory()
        >>>
        >>> tile_size = (1, 5, 4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath.name, tile_size)
        >>>
        >>>
        >>> p.get_labels() #we have 3 label classes
        [1, 2, 3]
        >>> p.remove_label(2) #remove label class 109
        True
        >>> p.get_labels() #only 2 classes remain selected
        [1, 3]

        '''
        if label not in self._labels:
            raise ValueError('not possible to remove label %s from label selection %s'
                             % (str(label), str(self._labels)))
        self._labels.remove(label)
        return True

    def add_label(self, label):
        '''
        Adds a label class to the selection.

        >>> from yapic_io.factories import make_tiff_interface
        >>> import tempfile
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
        >>> savepath = tempfile.TemporaryDirectory()
        >>>
        >>> tile_size = (1, 5, 4)
        >>> #upon object initialization all available image channels are set
        >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath.name, tile_size)
        >>>
        >>>
        >>> p.get_labels() #we have 3 label classes
        [1, 2, 3]
        >>> p.remove_label(2) #remove label class 2
        True
        >>> p.get_labels() #only 2 classes remain selected
        [1, 3]
        >>> p.remove_label(1) #remove label class 1
        True
        >>> p.get_labels() #only 1 class remains selected
        [3]
        >>> p.add_label(1)
        True
        >>> p.get_labels()
        [1, 3]
        >>> p.add_label(2)
        True
        >>> p.get_labels()
        [1, 2, 3]


        '''
        if label in self._labels:
            logger.warning('label class already selected %s', label)
            return False
        if label not in self._dataset.label_values():
            raise ValueError('not possible to add label class %s from dataset label classes %s'
                             % (str(label), str(self._dataset.label_values())))

        insort_left(self._labels, label)
        return True

    def _normalize(self, pixels):
        '''
        to be called by the self.pixels() function
        '''

        if self.normalize_mode is None:
            return pixels

        if self.normalize_mode == 'off':
            return pixels

        if self.normalize_mode == 'global':
            center_ref = np.array(self.global_norm_minmax[0])
            scale_ref = np.array(self.global_norm_minmax[1])

        if self.normalize_mode == 'local':
            # scale between 0 and 1
            #(data-minval)/(maxval-minval)
            # percentiles are used for minval and maxval to be robust
            # against outliers
            max_ref = np.percentile(pixels, 99, axis=(0, 2, 3, 4))
            min_ref = np.percentile(pixels, 1, axis=(0, 2, 3, 4))

            center_ref = min_ref
            scale_ref = max_ref - min_ref

        if self.normalize_mode == 'local_z_score':
             #(data-mean)/std
            center_ref = pixels.mean(axis=(0, 2, 3, 4))
            scale_ref = pixels.std(axis=(0, 2, 3, 4))

        pixels_centered = (pixels.swapaxes(1, 4) -
                           center_ref).swapaxes(1, 4)

        if (scale_ref == 0).all():
            # if scale_ref is zero, do not scale to avoid zero
            # division
            return pixels_centered
        # scale by scale reference
        return (pixels_centered.swapaxes(1, 4) /
                scale_ref).swapaxes(1, 4)
