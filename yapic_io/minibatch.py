import os
import logging
import numpy as np
from collections.abc import Iterable

logger = logging.getLogger(os.path.basename(__file__))


class Minibatch(object):
    '''
    Iterator providing pixel and label data for classifier training
    or prediction.
    This class is not directly used, but provides common functions
    as a parent class for TrainingBatch and PredictionBatch.

    Parameters
    ----------
    dataset: Dataset
        Handle for pixels and labels weights.
    batch_size: int
        Nr of tiles for one batch.
    size_zxy: (nr_zslices, nr_x, nr_y)
        3d tile size (size of classifier output template).
    padding_zxy: (z, x, y)
        Growing of pixel tile in z, x and y (to define size of classifier input
        template relative to output template size size_zxy).

    See Also
    --------
    yapic.io.training_batch
    yapic_io.prediction_batch
    '''
    def __init__(self, dataset, batch_size, size_zxy, padding_zxy=(0, 0, 0)):

        self.dataset = dataset

        self.pixel_dimension_order = [0, 1, 2, 3, 4]
        self.set_pixel_dimension_order('bczxy')

        self._batch_size = batch_size
        self.normalize_mode = None
        self.global_norm_minmax = None
        self.float_data_type = np.float32  # type of pixel and weight data

        if size_zxy:
            np.testing.assert_equal(len(size_zxy), 3,
                                    'len of size_zxy be 3: (z, x, y)')
        if padding_zxy:
            np.testing.assert_equal(len(padding_zxy), 3,
                                    'len of padding_shoule be 3: (z, x, y)')

        self.tile_size_zxy = size_zxy
        self.padding_zxy = padding_zxy

        # imports all available channels by default
        nr_channels = self.dataset.image_dimensions(0)[0]
        self.channels = set(range(nr_channels))
        # imports all available labels by default
        self.labels = set(self.dataset.label_values())

    def set_pixel_dimension_order(self, s):
        '''
        Parameters
        ----------
        s : {'bczxy', 'bzxyc'}
            Dimension order of pixel array as returned by self.pixels()
            and self.weights().
            First dimension is always the batch dimension and can not be
            changed.

        Notes
        -----
        Keras supports pixel dimension order 'bzxyc'.
        '''

        assert s in ['bczxy',
                     'bzxyc'], 'dimension order {} not supported'.format(s)

        self.pixel_dimension_order[0] = s.find('b')
        self.pixel_dimension_order[1] = s.find('c')
        self.pixel_dimension_order[2] = s.find('z')
        self.pixel_dimension_order[3] = s.find('x')
        self.pixel_dimension_order[4] = s.find('y')

        return self.pixel_dimension_order

    def set_tile_size(self, size_zxy):
        self.tile_size_zxy = size_zxy

    def set_normalize_mode(self, mode_str, minmax=None):
        '''
        Parameters
        ----------
        mode_str: {'local', 'local_z_score', 'global', 'minmax'}
            Normalization mode for pixel values. Channels are treated
            separately.

            * 'local': scale between 0 and 1 per minibatch.
            * 'local_z_score' : mean to zero, scale by standard deviation,
              calculated per minibatch
            * 'global': scale ``minmax=(min, max)`` between 0 and 1,
              calculated for whole dataset
        minmax : (min, max)
            Tuple of minimum and maximum pixel value for global normalization,
            e.g ``(0, 255)`` for 8 bit images, or list of tuples, one for each
            channel, e.g. ``[(0, 255), (0, 100), (0, 255)]``
        '''
        valid = ('local_z_score', 'local', 'off', 'global')
        msg = 'Invalid normalization mode: "{}". Valid choices: {}'
        assert mode_str in valid, msg.format(mode_str, valid)

        self.normalize_mode = mode_str

        if mode_str == 'global':

            if minmax is None:
                # calculate upper and lower percentile automatically
                minmax = self.dataset.pixel_statistics(self.channels)
            n_channels = self.dataset.pixel_connector.image_dimensions(0)[0]

            if len(minmax) == 2:
                if n_channels == 2:
                    assert (_is_twotuple_of_numerics(minmax)
                            or _is_list_of_twotuples(minmax))
                else:
                    assert _is_twotuple_of_numerics(minmax)
            else:
                assert n_channels == len(minmax)
                assert _is_list_of_twotuples(minmax)

            msg = 'Scale reference for pixel normalization must not be zero.'
            if _is_twotuple_of_numerics(minmax):
                assert minmax[1] != 0, msg
            if _is_list_of_twotuples(minmax):
                for e in minmax:
                    assert e[1] != 0, msg

            self.global_norm_minmax = minmax

    def _normalize(self, pixels):
        '''
        To be called by the ``self.pixels()`` function
        '''
        if self.normalize_mode in ('off', None):
            return pixels

        if self.normalize_mode == 'global':
            if _is_twotuple_of_numerics(self.global_norm_minmax):
                # same reference for all channels
                center_ref = np.array(self.global_norm_minmax[0])
                scale_ref = np.array(self.global_norm_minmax[1])
            else:
                center_ref = np.array([e[0] for e in self.global_norm_minmax])
                scale_ref = np.array([e[1] for e in self.global_norm_minmax])

        elif self.normalize_mode == 'local':
            # (data - minval) / (maxval - minval)
            # percentiles are used to be robust against outliers
            min_ref, max_ref = np.percentile(pixels,
                                             [1, 99],
                                             axis=(0, 2, 3, 4))

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


def _is_twotuple_of_numerics(tp):
    if not isinstance(tp, Iterable):
        return False
    if not len(tp) == 2:
        return False
    for e in tp:
        if not (isinstance(e, int) or isinstance(e, float)):
            return False
    return True


def _is_list_of_twotuples(ls):
    if not isinstance(ls, Iterable):
        return False
    for e in ls:
        if not _is_twotuple_of_numerics(e):
            return False
    return True
