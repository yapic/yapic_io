from bisect import insort_left
import os
import logging
import numpy as np
logger = logging.getLogger(os.path.basename(__file__))


class Minibatch(object):
    '''
    The selected labels (i.e. classes for prediction).
    Per default, all available labels are selected.

    The order of the labels list defines the order of the
    labels layer in the probability map (i.e. the classification result
    matrix) that can be exported with put_probmap_data().
    '''
    '''
    Returns the selected image channels.
    Per default, all available channels are selected.

    The order of the channels list defines the order of the
    channels layer in the pixels (accessed with get_pixels()).
    Pixels have the dimensionality (channels, z, x, y)

    >>> from yapic_io.factories import make_tiff_interface
    >>>
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/'
    >>> savepath = 'yapic_io/test_data/tmp/'
    >>>
    >>> tile_size = (1, 5, 4)
    >>> # make training_batch mb and prediction interface p: upon object initialization all available image channels are set
    >>> mb, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tile_size) #upon object initialization all available image channels are set
    >>>
    >>>
    >>> p.channel_list # we have 3 channels in the prediction interface
    [0, 1, 2]
    >>>
    >>> p[0].pixels().shape # accordingly, we have 3 channels in the 5D pixels tile (n, c, z, x, y)
    (10, 3, 1, 5, 4)
    >>>
    '''
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
        self.dataset = dataset
        self._batch_size = batch_size
        self.normalize_mode = None
        self.global_norm_minmax = None
        self.float_data_type = np.float32  # type of pixel and weight data

        if size_zxy:
            np.testing.assert_equal(len(size_zxy), 3, 'len of size_zxy be 3: (z, x, y)')
        if padding_zxy:
            np.testing.assert_equal(len(padding_zxy), 3, 'len of padding_shoule be 3: (z, x, y)')

        self.tile_size_zxy = size_zxy
        self.padding_zxy = padding_zxy

        # imports all available channels by default
        nr_channels = self.dataset.image_dimensions(0)[0]
        self.channels = set(range(nr_channels))
        # imports all available labels by default
        self.labels = set(self.dataset.label_values())

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
        valid = ('local_z_score', 'local', 'off', 'global')
        msg = 'Invalid normalization mode: "{}". Valid choices: {}'
        assert mode_str in valid, msg.format(mode_str, valid)

        self.normalize_mode = mode_str

        if mode_str == 'global':
            assert minmax is not None, 'normalization range (min, max) required'
            self.global_norm_minmax = minmax


    def _normalize(self, pixels):
        '''
        to be called by the self.pixels() function
        '''
        if self.normalize_mode in ('off', None):
            return pixels

        if self.normalize_mode == 'global':
            center_ref = np.array(self.global_norm_minmax[0])
            scale_ref = np.array(self.global_norm_minmax[1])
        elif self.normalize_mode == 'local':
            # (data - minval) / (maxval - minval)
            # percentiles are used to be robust against outliers
            max_ref = np.percentile(pixels, 99, axis=(0, 2, 3, 4))
            min_ref = np.percentile(pixels, 1, axis=(0, 2, 3, 4))

            center_ref = min_ref
            scale_ref = max_ref - min_ref
        elif self.normalize_mode == 'local_z_score':
             # (data - mean) / std
            center_ref = pixels.mean(axis=(0, 2, 3, 4))
            scale_ref = pixels.std(axis=(0, 2, 3, 4))

        pixels_centered = (pixels.swapaxes(1, 4) - center_ref)

        if (scale_ref == 0).all():
            # if scale_ref is zero, do not scale to avoid zero division
            return pixels_centered.swapaxes(1, 4)
        else:
            # scale by scale reference
            return (pixels_centered / scale_ref).swapaxes(1, 4)
