import numpy as np
from numpy.random import choice, randint
import random
import collections
import yapic_io.utils as ut
from functools import lru_cache
import logging
import os
import yapic_io.transformations as trafo
import sys

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
randint_array = np.vectorize(randint)
TrainingTile = collections.namedtuple('TrainingTile',
                                      ['pixels', 'channels', 'weights',
                                       'labels', 'augmentation'])


class Dataset(object):
    '''
    Provides connectors to pixel data source and assigned weights for
    classifier training.
    Provides methods for getting image tiles and data augmentation for
    classifier training, as well as writing classifier output tile-by-tile
    to target images.



    Parameters
    ----------
    pixel_connector : yapic_io.connector.Connector
        Connector object (e.g. TiffConnector or IlastikConnector) for binding
        of pixel and label data, as well as prediction result data.

    Notes
    -----
    Pixel data is loaded lazily to allow images of arbitrary size.
    Pixel data is cached in memory for repeated requests.
    '''

    def __init__(self, pixel_connector):

        self.pixel_connector = pixel_connector
        self.n_images = pixel_connector.image_count()
        self.label_counts = self.load_label_counts()

        # self.label_weights dict is complementary to self.label_counts
        self.label_weights = {label: 1 for label in self.label_counts.keys()}

        # max nr of trials to get a random training tile in polling mode
        self.max_pollings = 30

        is_consistent, channel_cnt = self.channels_are_consistent()
        msg = ('Varying number of channels: {}. '
               'Channel counts must be identical '
               'for all images in dataset. '
               'Dataset may be incomplete.').format(channel_cnt)
        assert is_consistent, msg

    def __repr__(self):
        return 'Dataset ({} images)'.format(self.n_images)

    @lru_cache(maxsize=1000)
    def image_dimensions(self, image_nr):
        '''
        Returns dimensions of the dataset.
        dims is a 4-element-tuple:

        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        (nr_channels, nr_zslices, nr_x, nr_y)
        '''
        return self.pixel_connector.image_dimensions(image_nr)

    def _smallest_image_size_xy(self):
        Z = 10
        X = 500
        Y = 500
        for i in range(self.n_images):
            size_z, size_x, size_y = self.image_dimensions(i)[-3:]
            if size_z < Z:
                Z = size_z
            if size_x < X:
                X = size_x
            if size_y < Y:
                Y = size_y
        return (Z, X, Y)

    def pixel_statistics(self,
                         channels,
                         upper=99,
                         lower=1,
                         tile_size_zxy=None,
                         n_tiles=1000):
        '''
        Performs random sampling of n tiles and calculates upper and lower
        percentiles of pixel values. Separated for each channel.
        This data can be used for normalization of pixel intensities.

        Parameters
        ----------
        channels : array_like
            List of pixel channels to be fetched.
        upper : float
            Upper percentile.
        lower : float
            Lower percentile.
        tile_size_zxy : (nr_zslices, nr_x, nr_y)
            Tile size.
        n_tiles : int
            Nr of tiles to be fetched

        Returns
        -------
        [(lower_01, upper_01), (lower_02, upper_02), (lower_03, upper_03), ...]
        '''

        if tile_size_zxy is None:
            tile_size_zxy = self._smallest_image_size_xy()
        percentiles = np.zeros((n_tiles, len(channels), 2))
        msg = ('\n\nCalculate global pixel statistics'
               '({} tiles of size {})...\n').format(n_tiles, tile_size_zxy)
        sys.stdout.write(msg)
        for i in range(n_tiles):
            image_nr, pos_zxy = self._random_pos_izxy(None, tile_size_zxy)
            pix = self.multichannel_pixel_tile(image_nr,
                                               pos_zxy,
                                               tile_size_zxy,
                                               channels)
            percentiles[i, :, 0] = np.percentile(pix, lower, axis=[1, 2, 3])
            percentiles[i, :, 1] = np.percentile(pix, upper, axis=[1, 2, 3])

        lower_bnds = np.min(percentiles[:, :, 0], axis=0)
        upper_bnds = np.max(percentiles[:, :, 1], axis=0)

        out = [(lower, upper) for lower, upper in zip(lower_bnds, upper_bnds)]
        msg = '{} and {} percentiles per channel: {}\n'.format(lower,
                                                               upper,
                                                               out)
        sys.stdout.write(msg)
        return out

    def channels_are_consistent(self):
        '''
        Returns True if all images of the dataset have the same number of
        channels. Otherwise False.

        Returns
        -------
        boolean
            True if channels are consistent
        list
            List with channel counts
        '''
        channel_cnt = np.unique([self.image_dimensions(i)[0]
                                for i in range(self.n_images)])

        if len(channel_cnt) == 1:
            return True, channel_cnt

        return False, channel_cnt

    def label_values(self):
        '''
        Get label values.

        Returns
        -------
        list
            Sorted list of label values.
        '''
        labels = list(self.label_counts.keys())
        labels.sort()
        return labels

    def random_training_tile(self,
                             size_zxy,
                             channels,
                             pixel_padding=(0, 0, 0),
                             equalized=False,
                             augment_params=None,
                             labels='all',
                             ensure_labelvalue=None):
        '''
        Returns a randomly chosen training tile including weights.

        Parameters
        ----------
        size_zxy : (nr_zslices, nr_x, nr_y)
            Tile size.
        channels : array_like
            List of pixel channels to be fetched.
        pixel_padding : (pad_z, pad_x, pad_y)
            Amount of padding to increase tile size in zxy.
        equalized : bool
            If true, less frequent label_values are picked with same
            probability as frequent label_values.
        augment_params : dict
            Image augmentation settings. Possible key:value pairs:
            fliplr : True or False;
            flipud : True or False;
            rot90 : int, number of 90 degree rotations;
            rotate : float, rotation in degrees;
            shear : float, shear in degrees.
        labels : array_like or str
            List of labelvalues to be fetched.
        ensure_labelvalue : int
            Labelvalue that must be present in the fetched tile region.

        Returns
        -------
        collections.namedtuple
            TrainingTile(pixels, channels, labels, weights, augmentation)
        '''
        augment_params = augment_params or {}
        if labels == 'all':
            labels = self.label_values()

        if hasattr(self.pixel_connector, 'label_index_to_coordinate'):
            # fetch by label index
            return self._random_training_tile_by_coordinate(
                size_zxy,
                channels,
                labels,
                pixel_padding=pixel_padding,
                equalized=equalized,
                augment_params=augment_params,
                ensure_labelvalue=ensure_labelvalue)
        else:
            return self._random_training_tile_by_polling(
                size_zxy,
                channels,
                labels,
                pixel_padding=pixel_padding,
                equalized=equalized,
                augment_params=augment_params,
                ensure_labelvalue=ensure_labelvalue)

    def _random_pos_izxy(self, label_value, tile_size_zxy):
        '''
        Get a random image and a random zxy position (for a tile of shape
        size_zxy) within this image.
        Images with more frequent labels of type label_value are more likely to
        be selected.
        '''
        # get random image by label probability
        # label probability per image
        label_prob = self._get_label_probs(label_value)
        img_nr = choice(len(label_prob), p=label_prob)

        # get random zxy position within selected image
        img_shape_zxy = self.image_dimensions(img_nr)[1:]
        img_maxpos_zxy = np.array(img_shape_zxy) - tile_size_zxy

        msg = 'Tile of size {} does not fit in image of size {}'.format(
                    tile_size_zxy, img_shape_zxy)
        assert (img_maxpos_zxy > -1).all(), msg

        pos_zxy = randint_array(img_maxpos_zxy + 1)
        return img_nr, pos_zxy

    def _get_label_probs(self, label_value):
        '''
        Get probabilities for labels per image if label_value is None,
        probabilities for the sum of all label values is returned.
        '''
        if label_value is None:
            # label count for all labelvalues
            lbl_count = sum(self.label_counts.values())
        else:
            lbl_count = self.label_counts[label_value]
        return lbl_count / lbl_count.sum()

    def _random_training_tile_by_polling(self,
                                         size_zxy,
                                         channels,
                                         labels,
                                         pixel_padding=(0, 0, 0),
                                         equalized=False,
                                         augment_params=None,
                                         ensure_labelvalue=None):
        ''''
        Fetches a random training tile by repeated polling until the label
        specified in ensure_labelvalue is at least one time represented in the
        tile. The number if trials is set in self.max_pollings.
        If the nr of trials exceeds max_pollings, the last fetched tile is
        returned, although not containing the label.
        '''
        augment_params = augment_params or {}
        if ensure_labelvalue is None and equalized:
            ensure_labelvalue = self._random_label_value(equalized=equalized)

        for counter in range(self.max_pollings):
            img_nr, pos_zxy = self._random_pos_izxy(ensure_labelvalue,
                                                    size_zxy)

            tile_data = self.training_tile(img_nr, pos_zxy, size_zxy,
                                           channels, labels,
                                           pixel_padding=pixel_padding,
                                           augment_params=augment_params)

            if ensure_labelvalue is None:
                # check if weights for any label are present
                are_weights_in_tile = tile_data.weights.any()
            else:
                # convert set to sorted array
                labels = np.array(sorted(tile_data.labels))
                # check if weights for specified label are present
                lblregion_index = np.where(labels == ensure_labelvalue)[0]
                weights_lblregion = tile_data.weights[lblregion_index]
                are_weights_in_tile = weights_lblregion.any()

            if are_weights_in_tile:
                msg = ('Needed {} trials to fetch random tile containing ' +
                       'labelvalue {}').format(counter,
                                               ensure_labelvalue or '<any>')
                if counter == 0:
                    logger.debug(msg)
                else:
                    logger.info(msg)
                return tile_data

        msg = ('Could not fetch random tile containing labelvalue {} ' +
               'within {} trials').format(ensure_labelvalue, counter)
        logger.warning(msg)

        # if no labelweights are present from any labelvalue
        if not tile_data.weights.any():
            msg = ('training labelweighs do not contain any weights above 0 ' +
                   'for any label')
            logger.warning(msg)

        return tile_data

    def _random_training_tile_by_coordinate(self,
                                            size_zxy,
                                            channels,
                                            labels,
                                            pixel_padding=(0, 0, 0),
                                            equalized=False,
                                            augment_params=None,
                                            ensure_labelvalue=None):
        augment_params = augment_params or {}
        if ensure_labelvalue is None:
            ensure_labelvalue = self._random_label_value(equalized=equalized)

        total_count = self.label_counts[ensure_labelvalue].sum()
        img_nr, _, *pos_zxy = self.label_coordinate(ensure_labelvalue,
                                                    randint(total_count))
        np.testing.assert_array_equal(len(pos_zxy), len(size_zxy))

        # Now we have a label. But we can but the tile anywhere as long as the
        # label is in it

        pos_zxy = np.array(pos_zxy)
        shape_zxy = np.array(self.image_dimensions(img_nr)[1:])

        maxpos = np.minimum(pos_zxy, shape_zxy - size_zxy)
        minpos = np.maximum(0, pos_zxy - size_zxy + 1)

        pos_zxy = [np.random.randint(a, b + 1) for a, b in zip(minpos, maxpos)]

        tile_data = self.training_tile(img_nr, pos_zxy, size_zxy,
                                       channels, labels,
                                       pixel_padding=pixel_padding,
                                       augment_params=augment_params)

        return tile_data

    def training_tile(self,
                      image_nr,
                      pos_zxy,
                      size_zxy,
                      channels,
                      labels,
                      pixel_padding=(0, 0, 0),
                      augment_params=None):
        '''
        Returns a training tile including weights.

        Parameters
        ----------
        image_nr : int
            Index of image.
        pos_zxy : (z, x, y)
            Upper left position of pixels in source image_nr.
        size_zxy : (nr_zslices, nr_x, nr_y)
            Tile size.
        channels : array_like
            List of pixel channels to be fetched.
        labels : array_like
            List of labelvalues to be fetched.
        pixel_padding : (pad_z, pad_x, pad_y)
            Amount of padding to increase tile size in zxy.
        augment_params : dict
            Image augmentation settings. Possible key:value pairs:
            fliplr : True or False;
            flipud : True or False;
            rot90 : int, number of 90 degree rotations;
            rotate : float, rotation in degrees;
            shear : float, shear in degrees.

        Returns
        -------
        collections.namedtuple
            TrainingTile(pixels, channels, labels, weights, augmentation)
        '''
        augment_params = augment_params or {}
        # 4d pixel tile with selected channels in 1st dimension
        pixel_tile = self.multichannel_pixel_tile(
                        image_nr, pos_zxy, size_zxy, channels,
                        pixel_padding=pixel_padding,
                        augment_params=augment_params)

        # 4d label tile with selected labels in 1st dimension
        shape_zxy = self.image_dimensions(image_nr)[1:]
        label_tile = [_augment_tile(shape_zxy, pos_zxy, size_zxy,
                                    self._get_weights_tile,
                                    augment_params=augment_params,
                                    image_nr=image_nr,
                                    label_value=l)
                      for l in labels]
        label_tile = np.array(label_tile)

        msg = 'pixel tile dim={} label tile dim={} labels={}'.format(
                    pixel_tile.shape, label_tile.shape, len(labels))
        logger.debug(msg)

        return TrainingTile(pixel_tile, channels, label_tile, labels,
                            augment_params)

    def multichannel_pixel_tile(self,
                                image_nr,
                                pos_zxy,
                                size_zxy,
                                channels,
                                pixel_padding=(0, 0, 0),
                                augment_params=None):
        augment_params = augment_params or {}
        np.testing.assert_equal(len(pos_zxy), 3,
                                'Expected 3 dimensions (Z, X, Y)')
        np.testing.assert_equal(len(size_zxy), 3,
                                'Expected 3 dimensions (Z, X, Y)')

        image_shape_zxy = self.image_dimensions(image_nr)
        ut.assert_valid_image_subset(image_shape_zxy[1:], pos_zxy, size_zxy)

        pixel_padding = np.array(pixel_padding)
        size_padded = size_zxy + 2 * pixel_padding
        pos_padded = pos_zxy - pixel_padding

        for c in channels:
            msg = 'channel {} does not exist'.format(c)
            assert c < self.pixel_connector.image_dimensions(image_nr)[0], msg
        tile = [_augment_tile(image_shape_zxy,
                              np.hstack([[c], pos_padded]),
                              np.hstack([[1], size_padded]),
                              self.pixel_connector.get_tile,
                              augment_params=augment_params,
                              image_nr=image_nr)
                for c in channels]

        return np.vstack(tile)

    def _get_weights_tile(self, image_nr=None, pos=None, size=None,
                          label_value=None):
        '''
        Returns a 3d weight matrix tile for a certain label with
        dimensions zxy.
        '''
        assert label_value in self.label_values()

        boolmat = self.pixel_connector.label_tile(image_nr, pos, size,
                                                  label_value)

        weight_mat = np.zeros_like(boolmat, float)
        weight_mat[boolmat] = self.label_weights[label_value]

        return weight_mat

    def equalize_label_weights(self):
        '''
        equalizes labels according to their amount.
        less frequent labels are weighted higher than more frequent labels
        '''

        weights = {
            l: np.nan_to_num(1.0 / img_counts.sum())
            for l, img_counts in self.label_counts.items()
        }

        self.label_weights = {l: c / sum(weights.values())
                              for l, c in weights.items()}

    def load_label_counts(self):
        '''
        Returns the cout of each labelvalue for each image as dict

        Returns
        -------
        dict
            Keys are labelvalues. Values are lists with label counts for each
            image.

        Examples
        --------
        >>> from yapic_io.dataset import Dataset
        >>> from yapic_io.tiff_connector import TiffConnector
        >>> from pprint import pprint
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
        >>>
        >>> c = TiffConnector(pixel_image_dir, label_image_dir)
        >>> d = Dataset(c)
        >>> d.load_label_counts()
        {1: array([4, 0, 0]), 2: array([ 3,  0, 11]), 3: array([3, 0, 3])}
        '''

        logger.debug('Starting to load label counts...')

        label_counts = collections.defaultdict(lambda: np.zeros(self.n_images,
                                                                dtype='int64'))
        for i in range(self.n_images):
            img_label_counts = \
                self.pixel_connector.label_count_for_image(i) or {}

            for label_value in img_label_counts.keys():
                label_counts[label_value][i] = img_label_counts[label_value]

        return dict(label_counts)

    def sync_label_counts(self, datset):
        '''
        Should be applied e.g. if two datasets were created from splitted
        conncetors( e.g. with tiff_connector.split(). Typical use case
        is syncing a training dataset and a validation dataset.

        Parameters
        ----------
        datset : Dataset
             Dataset to sync labeldata.

        Notes
        -----
        When splitting Connectors it may occur that one or more label values
        are only present in one Connector. Sync_label_counts adds missing
        label values to a dataset. Note that the missing label values will
        have a label count of 0 after syncing.

        Examples
        --------
        Split one TiffConnector in two and sync label data.

        >>> from yapic_io.dataset import Dataset
        >>> from yapic_io.tiff_connector import TiffConnector
        >>> from pprint import pprint
        >>>
        >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
        >>> c = TiffConnector(pixel_image_dir, label_image_dir)
        >>> c1, c2 = c.split(1./3.)
        >>>
        >>> d = Dataset(c)
        >>> d1 = Dataset(c1)
        >>> d2 = Dataset(c2)
        >>>
        >>> pprint(d.label_counts) # full dataset has three labelvalues
        {1: array([4, 0, 0]), 2: array([ 3,  0, 11]), 3: array([3, 0, 3])}
        >>>
        >>> pprint(d1.label_counts) # d1 has also three labelvalues
        {1: array([4]), 2: array([3]), 3: array([3])}
        >>>
        >>> pprint(d2.label_counts) #d2 has only two labelvalues
        {2: array([ 0, 11]), 3: array([0, 3])}
        >>>
        >>>
        >>> d1.sync_label_counts(d2)
        >>> pprint(d1.label_counts)
        {1: array([4]), 2: array([3]), 3: array([3])}
        >>>
        >>> pprint(d2.label_counts) #labelvalue 1 added to d2 (with count 0)
        {1: array([0, 0]), 2: array([ 0, 11]), 3: array([0, 3])}

        '''
        lc1 = self.label_counts
        lc2 = datset.label_counts

        # missing label in each dataset
        missing_in_d1 = set(lc2.keys()) - set(lc1.keys())
        missing_in_d2 = set(lc1.keys()) - set(lc2.keys())

        for lbl in missing_in_d1:
            lc1[lbl] = np.zeros((self.n_images), dtype=np.int64)
        for lbl in missing_in_d2:
            lc2[lbl] = np.zeros((datset.n_images), dtype=np.int64)

    def _random_label_value(self, equalized=False):
        '''
        Returns a randomly chosen labelvalue.

        Parameters
        ----------
        equalized : bool, optional
            If true, less frequent label_values are picked with same
            probability as frequent label_values.
        '''
        if equalized:
            return random.choice(self.label_values())

        # probabilities for each labelvalue
        total_counts = [counts.sum() for counts in self.label_counts.values()]
        total_counts_norm = np.array(total_counts) / sum(total_counts)

        # pick a labelvalue according to the labelvalue probability
        return np.random.choice(self.label_values(), p=total_counts_norm)


def inner_tile_size(image_shape, pos, tile_shape):
    '''
    If a requested tile is out of bounds, this function calculates a transient
    tile position size and pos. The transient tile has to be padded in a later
    step to extend the edges for delivering the originally requested out of
    bounds tile. The padding sizes for this step are also delivered.
    size_out and pos_out that can be used in a second step with padding.
    pos_tile defines the position in the transient tile for cutting out the
    originally requested tile.

    Paramters
    ---------
    image_shape: array_like
        shape of full size original image
    pos: array_like
        Upper left position of tile in full size original image
    tile_shape: array_like
        Size of tile.

    Returns
    -------
    pos_out, size_out, pos_tile, padding_sizes
        pos_out is the position inside the full size original image for the
        transient tile.
        (more explanation needed)
    '''
    padding_lower = np.maximum(0, -pos)
    padding_upper = np.maximum(0, pos - (image_shape - tile_shape))
    padding = np.vstack([padding_lower, padding_upper]).T

    pos_tile = np.maximum(0, pos + padding_upper - image_shape)
    pos_out = pos + padding_lower - pos_tile

    # WTF is going on here??
    size_inmat = tile_shape + np.minimum(0, pos)

    size_new_1 = np.maximum(padding_lower, size_inmat)
    size_new_2 = np.minimum(tile_shape, image_shape - pos_out)
    size_out = np.minimum(size_new_1, size_new_2)

    return tuple(pos_out), tuple(size_out), tuple(pos_tile), tuple(padding)


def _augment_tile(img_shape,
                  pos,
                  tile_shape,
                  get_tile_func,
                  augment_params=None,
                  **kwargs):
    '''
    fetch tile and augment it
    if rotation and shear is activated, a 3 times larger tile
    is fetched and the final tile is cut out from that after
    rotation/shear.
    '''
    augment_params = augment_params or {}
    rotation_angle = augment_params.get('rotation_angle', 0)
    shear_angle = augment_params.get('shear_angle', 0)

    pos = np.array(pos)
    orig_tile_shape = np.array(tile_shape)
    tile_shape = np.array(tile_shape)

    augment_fast = (tile_shape[-2:] > 1).any()
    augment_slow = augment_fast and (rotation_angle > 0 or shear_angle > 0)

    if augment_slow:
        pos -= tile_shape
        tile_shape *= 3

    res = inner_tile_size(img_shape, pos, tile_shape)
    pos_transient, size_transient, pos_inside_transient, pad_size = res

    tile = get_tile_func(pos=pos_transient, size=size_transient, **kwargs)

    tile = np.pad(tile, pad_size, mode='symmetric')
    mesh = ut.get_tile_meshgrid(tile.shape, pos_inside_transient, tile_shape)
    tile = tile[tuple(mesh)]

    if augment_fast:
        # if the requested tile is only of size 1 in x and y,
        # augmentation can be omitted, since rotation and flipping always
        # occurs around the center axis.
        rot90 = augment_params.get('rot90', 0)
        flipud = augment_params.get('flipud', False)
        fliplr = augment_params.get('fliplr', False)

        tile = trafo.flip_image_2d_stack(tile, fliplr=fliplr,
                                         flipud=flipud, rot90=rot90)

    if augment_slow:
        tile = trafo.warp_image_2d_stack(tile, rotation_angle, shear_angle)
        mesh = ut.get_tile_meshgrid(tile.shape, orig_tile_shape,
                                    orig_tile_shape)
        tile = tile[tuple(mesh)]

    return tile
