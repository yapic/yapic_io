import numpy as np
from numpy.random import choice, randint
randint_array = np.vectorize(randint)

import random

import yapic_io.utils as ut
from functools import lru_cache
import logging
import os
import yapic_io.transformations as trafo
import collections
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.INFO)
TrainingTile = collections.namedtuple('TrainingTile',
                                      ['pixels', 'channels', 'weights', 'labels', 'augmentation'])


class Dataset(object):
    '''
    provides connectors to pixel data source and
    (optionally) assigned weights for classifier training

    provides methods for getting image tiles and data
    augmentation for training

    pixel data is loaded lazily to allow images of arbitrary size
    pixel data is cached in memory for repeated requests
    '''

    def __init__(self, pixel_connector):
        self.pixel_connector = pixel_connector
        self.n_images = pixel_connector.image_count()
        self.label_counts = self.load_label_counts()
        self.init_label_weights()
        self.max_pollings = 30  # max nr of trials to get a
        # random training tile in polling mode

    def __repr__(self):
        return 'Dataset (%s images)' % (self.n_images)

    @lru_cache(maxsize=1000)
    def image_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:

        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''
        return self.pixel_connector.image_dimensions(image_nr)

    def channel_list(self):
        nr_channels = self.image_dimensions(0)[0]
        return list(range(nr_channels))

    def label_values(self):
        labels = list(self.label_counts.keys())
        labels.sort()
        return labels

    def put_prediction_tile(self, probmap_tile, pos_zxy, image_nr, label_value):
        # check if pos and tile size are 3d
        if not self.is_image_nr_valid(image_nr):
            raise ValueError('no valid image nr: %s' % str(image_nr))

        if not _check_pos_size(pos_zxy, probmap_tile.shape, 3):
            raise ValueError('pos, size or shape have %s dimensions instead of 3'
                             % str(len(pos_zxy)))

        return self.pixel_connector.put_tile(probmap_tile, pos_zxy, image_nr, label_value)

    def random_training_tile(self,
                             size_zxy,
                             channels,
                             pixel_padding=(0, 0, 0),
                             equalized=False,
                             augment_params=None,
                             labels='all',
                             label_region=None):
        '''
        returns a randomly chosen training tile that contains labels
        of specific labelvalue specified in label_region.

        :param pos_zxy: tuple defining the upper left position of the tile in 3 dimensions zxy
        :type pos_zxy: tuple
        :param channels: list of pixel channels to be fetched
        :param pixel_padding: amount of padding to increase tile size in zxy
        :param equalized: If true, less frequent label_values are picked with same
                          probability as frequent label_values
        :param labels: list of labelvalues to be fetched
        :param label_region: labelvalue that must be present in the fetched tile region                   

        :returns: TrainingTile (pixels and corresponding labels)
        '''
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
                label_region=label_region)

        return self._random_training_tile_by_polling(
            size_zxy,
            channels,
            labels,
            pixel_padding=pixel_padding,
            equalized=equalized,
            augment_params=augment_params,
            label_region=label_region)

    def _random_pos_izxy(self, label_value, size_zxy):
        '''
        get a random image and a random zxy position 
        (for a tile of shape size_zxy) within this image.
        Images with more frequent labels of type label_value
        are more likely to be selected.  
        '''

        # get random image by label probability
        # label probability per image
        label_prob = self._get_label_probs(label_value)
        img_nr_sel = choice(range(0, len(label_prob)), p=label_prob)

        # get random zxy position within selected image
        img_shape_zxy = self.image_dimensions(img_nr_sel)[1:]
        img_maxpos_zxy = ut.get_max_pos_for_tile(size_zxy, img_shape_zxy)

        err_msg = 'Tile of size {} does not fit in image of size {}'.format(
            size_zxy, img_shape_zxy)
        if (img_maxpos_zxy < 0).any():
            raise ValueError(err_msg)

        pos_zxy_sel = randint_array((0, 0, 0), img_maxpos_zxy + 1)
        pos_izxy_sel = np.insert(pos_zxy_sel, 0, img_nr_sel)

        return pos_izxy_sel

    def _get_label_probs(self, label_value):
        '''
        get probabilities for labels per image
        if label_value is None, probabilities for 
        the sum of all label values is returned.
        '''
        if label_value is None:
            #label count for all labelvalues
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
                                         label_region=None):
        ''''
        fetches a random training tile by repeated polling until
        the label specified in label_region is at least one time
        represented in the tile. The number if trials is set in
        self.max_pollings.
        If the nr of trials exceeds max_pollings, the last fetched
        tile is returned, although not containing the label.
        '''
        if label_region is None and equalized:
            label_region = self.random_label_value(equalized=equalized)

        counter = 0
        for i in range(self.max_pollings):
            counter += 1

            pos_izxy = self._random_pos_izxy(label_region, size_zxy)
            img_nr = pos_izxy[0]
            pos_zxy = pos_izxy[1:]

            tile_data = self.training_tile(img_nr, pos_zxy, size_zxy,
                                           channels, labels,
                                           pixel_padding=pixel_padding,
                                           augment_params=augment_params)
            if label_region is None:
                #check if weights for any label are present
                are_weights_in_tile = tile_data.weights.any() 
            else:
                #check if weights for specified label are present    
                lblregion_index = np.where(
                    np.array(tile_data.labels) == label_region)[0]
                
                weights_lblregion = tile_data.weights[lblregion_index]
                are_weights_in_tile = weights_lblregion.any()
           
            if are_weights_in_tile:
                if label_region:
                    msg = 'neened {} trials to fetch random tile containing labelvalue {}'.format(
                                counter, label_region)
                    logger.info(msg)
                else:
                    msg = 'neened {} trials to fetch random tile containing any labelvalue'.format(
                                counter)
                    logger.info(msg)       
                return tile_data
        msg = 'Could not fetch random tile containing labelvalue {} within {} trials'.format(
                            label_region, counter)
        logger.warning(msg)
        
        #if no labelweights are present from any labelvalue
        if not tile_data.weights.any():
            logger.warning('training labelweighs do not contain any weights above 0 for any label')

            
        return tile_data    

    def _random_training_tile_by_coordinate(self,
                                            size_zxy,
                                            channels,
                                            labels,
                                            pixel_padding=(0, 0, 0),
                                            equalized=False,
                                            augment_params=None,
                                            label_region=None):
        if label_region is None:
            # pick training tile where it is assured that weights for a specified label
            # are within the tile. the specified label is label_region
            _, coor = self.random_label_coordinate(equalized=equalized)
        else:
            _, coor = self.random_label_coordinate_for_label(label_region)

        img_nr = coor[0]
        coor_zxy = coor[2:]
        shape_zxy = self.image_dimensions(img_nr)[1:]
        pos_zxy = np.array(ut.get_random_pos_for_coordinate(
            coor_zxy, size_zxy, shape_zxy))

        tile_data = self.training_tile(img_nr, pos_zxy, size_zxy,
                                       channels, labels, pixel_padding=pixel_padding,
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
        # 4d pixel tile with selected channels in 1st dimension
        pixel_tile = self.multichannel_pixel_tile(image_nr, pos_zxy, size_zxy, channels,
                                                  pixel_padding=pixel_padding,
                                                  augment_params=augment_params)

        label_tile = []
        for label in labels:
            tile = self.label_tile(image_nr, pos_zxy, size_zxy, label,
                                   augment_params=augment_params)
            label_tile.append(tile)
        # 4d label tile with selected labels in 1st dimension
        label_tile = np.array(label_tile)

        logger.debug('pixel tile dim={} label tile dim={} labels={}'.format(
            pixel_tile.shape, label_tile.shape, len(labels)))
        augmentation = augment_params
        return TrainingTile(pixel_tile, channels, label_tile, labels, augmentation)

    def multichannel_pixel_tile(self,
                                image_nr,
                                pos_zxy,
                                size_zxy,
                                channels,
                                pixel_padding=(0, 0, 0),
                                augment_params=None):
        if not _check_pos_size(pos_zxy, size_zxy, 3):
            logger.debug('checked pos size in multichannel_pixel_tile')
            raise ValueError('pos and size must have length 3. pos_zxy: %s, size_zxy: %s'
                             % (str(pos_zxy), str(size_zxy)))

        image_shape_zxy = self.image_dimensions(image_nr)[1:]
        if not ut.is_valid_image_subset(image_shape_zxy, pos_zxy, size_zxy):
            raise ValueError('image subset not correct')

        pixel_padding = np.array(pixel_padding)

        size_padded = size_zxy + 2 * pixel_padding
        pos_padded = pos_zxy - pixel_padding

        pixel_tile = []
        for channel in channels:
            tile = self.tile_singlechannel(image_nr, tuple(pos_padded), tuple(size_padded), channel,
                                           augment_params=augment_params)
            pixel_tile.append(tile)

        # 4d pixel tile with selected channels in 1st dimension
        pixel_tile = np.array(pixel_tile)

        return pixel_tile

    def tile_singlechannel(self,
                           image_nr,
                           pos_zxy,
                           size_zxy,
                           channel,
                           reflect=True,
                           augment_params=None):
        '''
        returns a recangular subsection of an image with specified size.
        if requested tile is out of bounds, values will be added by reflection

        :param image_nr: image index
        :type image_nr: int
        :param pos_zxy: tuple defining the upper left position of the tile in 3 dimensions zxy
        :type pos_zxy: tuple
        :param size_zxy: tuple defining size of tile in 3 dimensions zxy
        :type size_zxy: tuple
        :returns: 3d tile as numpy array with dimensions zxy
        '''
        if not self.is_image_nr_valid(image_nr):
            return False

        if not _check_pos_size(pos_zxy, size_zxy, 3):
            return False

        pos_czxy = np.array([channel] + list(pos_zxy))
        # size in channel dimension is set to 1 to only select a single channel
        size_czxy = np.array([1] + list(size_zxy))

        shape_czxy = self.image_dimensions(image_nr)

        tile = augment_tile(shape_czxy,
                            pos_czxy,
                            size_czxy,
                            self.pixel_connector.get_tile,
                            augment_params=augment_params,
                            reflect=reflect,
                            **{'image_nr': image_nr})

        return np.squeeze(tile, axis=(0, ))

    def label_tile(self,
                   image_nr,
                   pos_zxy,
                   size_zxy,
                   label_value,
                   reflect=True,
                   augment_params=None):
        '''
        returns a recangular subsection of label weights with specified size.
        if requested tile is out of bounds, values will be added by reflection

        :param image_nr: image index
        :type image: int
        :param pos_zxy: tuple defining the upper left position of the tile in 3 dimensions zxy
        :type pos_zxy: tuple
        :param size_zxy: tuple defining size of tile in 3 dimensions zxy
        :type size: tuple
        :param label_value: label identifier
        :type label_value: int
        :returns: 3d tile of label weights as numpy array with dimensions zxy
        '''
        if not _check_pos_size(pos_zxy, size_zxy, 3):
            raise ValueError('pos, size or shape have %s dimensions instead of 3'
                             % len(pos_zxy))

        shape_zxy = self.image_dimensions(image_nr)[1:]

        tile = augment_tile(shape_zxy,
                            pos_zxy,
                            size_zxy,
                            self._label_tile_inner,
                            augment_params=augment_params,
                            reflect=reflect,
                            **{'image_nr': image_nr, 'label_value': label_value})

        return tile

    def _label_tile_inner(self, image_nr=None, pos=None, size=None, label_value=None):
        '''
        returns a 3d weight matrix tile for a ceratin label with dimensions zxy.
        '''
        if not self.is_label_value_valid(label_value):
            return False

        pos_zxy = pos
        size_zxy = size
        label_weight = self.label_weights[label_value]

        boolmat = self.pixel_connector.label_tile(image_nr,
                                                  pos_zxy, size_zxy, label_value)

        weight_mat = np.zeros(boolmat.shape)
        weight_mat[boolmat] = label_weight

        return weight_mat

    def set_label_weight(self, weight, label_value):
        '''
        sets the same weight for all labels of label_value
        in self.label_weights
        :param weight: weight value
        :param label_value: label
        '''
        if not self.is_label_value_valid(label_value):
            logger.warning(
                'could not set label weight for label value %s', str(label_value))
            return False

        self.label_weights[label_value] = weight
        return True

    def init_label_weights(self):
        '''
        Inits a dictionary with label weights. All weights are set to 1.
        the self.label_weights dict is complementary to self.label_counts. It defines
        a weight for each label.

        {
            label_nr1 : weight
            label_nr2 : weight
        }
        '''
        weight_dict = {}
        label_values = self.label_counts.keys()
        for label in label_values:
            weight_dict[label] = 1

        self.label_weights = weight_dict
        return True

    def equalize_label_weights(self):
        '''
        equalizes labels according to their amount.
        less frequent labels are weighted higher than more frequent labels
        '''
        labels = self.label_weights.keys()
        total_label_count = dict.fromkeys(labels)

        for label in labels:
            total_label_count[label] = self.label_counts[label].sum()

        self.label_weights = equalize_label_weights(total_label_count)
        return True

    def load_label_counts(self):
        '''
        returns the cout of each labelvalue for each image as dict

        label_counts = {
             label_value_1 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_2 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             label_value_2 : [nr_labels_image_0, nr_labels_image_1, nr_labels_image_2, ...],
             ...
        }
        '''
        logger.debug('start loading label counts...')
        label_counts_raw = [self.pixel_connector.label_count_for_image(im)
                            for im in range(self.n_images)]

        # identify all label_values in dataset
        label_values = []
        for label_count in label_counts_raw:
            if label_count is not None:
                label_values += label_count.keys()
        label_values = set(label_values)

        # init empty label_counts dict
        label_counts = {key: np.zeros(self.n_images, dtype='int64') for key in label_values}

        for i in range(self.n_images):
            if label_counts_raw[i] is not None:
                for label_value in label_counts_raw[i].keys():
                    label_counts[label_value][
                        i] = label_counts_raw[i][label_value]

        logger.debug('label_counts:')
        logger.debug(label_counts)
        return label_counts

    
    def sync_label_counts(self, datset):
        '''
        Should be applied e.g. if two datasets were created from splitted
        conncetors( e.g. with tiff_connector.split(). Typical use case
        is syncing a training dataset and a validation dataset.

        >>> from yapic_io.dataset import Dataset
        >>> from yapic_io.tiff_connector import TiffConnector
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
        >>> d.label_counts # full dataset has three labelvalues
        {1: array([4, 0, 0]), 2: array([ 3,  0, 11]), 3: array([3, 0, 3])}
        >>>
        >>> d1.label_counts # d1 has also three labelvalues
        {1: array([4]), 2: array([3]), 3: array([3])}
        >>>
        >>> d2.label_counts #d2 has only two labelvalues
        {2: array([ 0, 11]), 3: array([0, 3])}
        >>>
        >>>
        >>> d1.sync_label_counts(d2)
        >>> d1.label_counts
        {1: array([4]), 2: array([3]), 3: array([3])}
        >>>
        >>> d2.label_counts #labelvalue 1 added to d2 (with count 0)
        {2: array([ 0, 11]), 3: array([0, 3]), 1: array([0, 0])}
        
        '''  
        lc1 = self.label_counts
        lc2 = datset.label_counts

        #missing label in each dataset
        missing_in_d1 = set(lc2.keys()) - set(lc1.keys())
        missing_in_d2 = set(lc1.keys()) - set(lc2.keys())
        

        for lbl in missing_in_d1:
            lc1[lbl] = np.zeros((self.n_images), dtype=np.int64)
        for lbl in missing_in_d2:
            lc2[lbl] = np.zeros((datset.n_images), dtype=np.int64)
        







    def is_label_value_valid(self, label_value):
        '''
        check if label value is part of self.label_coordinates
        '''
        if label_value not in self.label_counts.keys():
            logger.warning('Label value not found %s', str(label_value))
            return False
        return True

    def is_image_nr_valid(self, image_nr):
        '''
        check if image_nr is between 0 and self.n_images
        '''
        if (image_nr >= self.n_images) or (image_nr < 0):
            msg = 'Wrong image number. image numbers in range 0 to %s'
            logger.error(msg, str(self.n_images - 1))
            return False
        return True

    def label_coordinate(self, label_value, label_index):
        '''
        for each labelvalue there exist `index` labels in the dataset.
        returns the coordinate (image_nr, channel, z, x, y) of the nth label.
        '''
        counts = self.label_counts[label_value]
        counts_cs = counts.cumsum()
        total_count = counts_cs[-1]

        np.testing.assert_array_less(-1, label_index)
        np.testing.assert_array_less(label_index, total_count)

        image_nr = np.argmax(counts_cs > label_index)
        if image_nr > 0:
            label_index -= counts_cs[image_nr - 1]

        coor_czxy = self.pixel_connector.label_index_to_coordinate(image_nr,
                                                                   label_value, label_index)
        coor_iczxy = np.insert(coor_czxy, 0, image_nr)
        return coor_iczxy

    def random_label_coordinate_for_label(self, label_value):
        '''
        returns a rondomly chosen label coordinate and the label value for a givel label value:

        (label_value, (img_nr, channel, z, x, y))

        channel is always zero!!

        :param equalized: If true, less frequent label_values are picked with same probability as frequent label_values
        :type equalized: bool
        '''
        if not self.is_label_value_valid(label_value):
            raise ValueError('label value %s not valid, possible label values are %s'
                             % (str(label_value), str(self.label_values())))

        counts = self.label_counts[label_value]
        total_count = counts.sum()

        if total_count < 1:  # if no labels of that value
            raise ValueError('no labels of value %s existing' %
                             str(label_value))
        choice = random.randint(0, total_count - 1)

        return (label_value, self.label_coordinate(label_value, choice))

    def random_label_value(self, equalized=False):
        '''
        returns a randomly chosen label value:

        :param equalized: If true, less frequent label_values are picked with same probability as frequent label_values
        :type equalized: bool
        '''
        labels = self.label_values()
        if equalized:
            chosen_label = random.choice(labels)
            return chosen_label

        label_values = []
        total_counts = []
        for label_value in self.label_counts.keys():
            label_values.append(label_value)
            total_counts.append(self.label_counts[label_value].sum())

        # probabilities for each labelvalue
        total_counts_norm = np.array(total_counts) / sum(total_counts)
        total_counts_norm_cs = total_counts_norm.cumsum()
        label_values = np.array(label_values)

        # pick a labelvalue according to the labelvalue probability
        random_nr = random.uniform(0, 1)
        chosen_label = label_values[
            (random_nr <= total_counts_norm_cs).nonzero()[0][0]]

        return chosen_label

    def random_label_coordinate(self, equalized=False):
        '''
        returns a randomly chosen label coordinate and the label value:

        (label_value, (img_nr, channel, z, x, y))

        channel is always zero!!

        :param equalized: If true, less frequent label_values are picked with same probability as frequent label_values
        :type equalized: bool
        '''

        chosen_label = self.random_label_value(equalized=equalized)
        return self.random_label_coordinate_for_label(chosen_label)
        # labels = self.label_values()
        # if equalized:
        #     label_sel = random.choice(labels)
        #     return self.random_label_coordinate_for_label(label_sel)
        # else:
        #     label_values = []
        #     total_counts = []
        #     for label_value in self.label_counts.keys():
        #         label_values.append(label_value)
        #         total_counts.append(self.label_counts[label_value].sum())

        #     # probabilities for each labelvalue
        #     total_counts_norm = np.array(total_counts)/sum(total_counts)
        #     total_counts_norm_cs = total_counts_norm.cumsum()
        #     label_values = np.array(label_values)

        #     # pick a labelvalue according to the labelvalue probability
        #     random_nr = random.uniform(0, 1)
        #     chosen_label = label_values[(random_nr <= total_counts_norm_cs).nonzero()[0][0]]

        #     return self.random_label_coordinate_for_label(chosen_label)


def label_values(self):
    return set(self.label_counts.keys())


def get_padding_size(shape, pos, size):
    '''
    [(x_lower, x_upper), (y_lower, _y_upper)]
    '''
    padding_size = []
    for sh, po, si in zip(shape, pos, size):
        p_l = 0
        p_u = 0
        if po < 0:
            p_l = abs(po)
        if po + si > sh:
            p_u = po + si - sh
        padding_size.append((p_l, p_u))
    return padding_size


def is_padding(padding_size):
    '''
    check if are all zero (False) or at least one is not zero (True)
    '''
    for dim in padding_size:
        if np.array(dim).any():  # if any nonzero element
            return True
    return False


def inner_tile_size(shape, pos, size):
    '''
    if a requested tile is out of bounds, this function calculates
    a transient tile position size and pos. The transient tile has to be padded in a
    later step to extend the edges for delivering the originally requested out of
    bounds tile. The padding sizes for this step are also delivered.
    size_out and pos_out that can be used in a second step with padding.
    pos_tile defines the position in the transient tile for cuuting out the originally
    requested tile.

    :param shape: shape of full size original image
    :param pos: upper left position of tile in full size original image
    :param size: size of tile
    :returns pos_out, size_out, pos_tile, padding_sizes

    pos_out is the position inside the full size original image for the transient tile.
    (more explanation needed)
    '''
    shape = np.array(shape)
    pos = np.array(pos)
    size = np.array(size)

    padding_sizes = get_padding_size(shape, pos, size)
    padding_upper = np.array([e[1] for e in padding_sizes])
    padding_lower = np.array([e[0] for e in padding_sizes])

    shift_1 = padding_lower
    shift_2 = shape - pos - padding_upper
    shift_2[shift_2 > 0] = 0

    shift = shift_1 + shift_2
    pos_tile = -shift + padding_lower
    pos_out = pos + shift

    dist_lu_s = shape - pos - shift
    size_new_1 = np.vstack([size, dist_lu_s]).min(axis=0)
    pos_r = pos.copy()
    pos_r[pos > 0] = 0
    size_inmat = size + pos_r

    size_new_2 = np.vstack([padding_lower, size_inmat]).max(axis=0)
    size_out = np.vstack([size_new_1, size_new_2]).min(axis=0)

    return tuple(pos_out), tuple(size_out), tuple(pos_tile), padding_sizes


def distance_to_upper_img_edge(shape, pos):
    return np.array(shape) - np.array(pos) - 1


def pos_shift_for_padding(shape, pos, size):
    padding_size = get_padding_size(shape, pos, size)
    dist = distance_to_upper_img_edge(shape, pos)

    return dist + 1 - padding_size


def equalize_label_weights(label_n):
    '''
    :param label_n: dict with label numbers where keys are label_values
    :returns dict with equalized weights for each label value
    label_n = {
            label_value_1 : n_labels,
            label_value_2 : n_labels,
            ...
            }
    '''
    nn = 0
    labels = label_n.keys()
    for label in labels:
        nn += label_n[label]

    weight_total_per_labelvalue = float(nn) / float(len(labels))

    # equalize
    eq_weight = {}
    eq_weight_total = 0
    for label in labels:
        eq_weight[label] = \
            weight_total_per_labelvalue / float(label_n[label])
        eq_weight_total += eq_weight[label]

    # normalize
    for label in labels:
        eq_weight[label] = eq_weight[label] / eq_weight_total

    return eq_weight


def augment_tile(shape,
                 pos,
                 size,
                 get_tile_func,
                 augment_params=None,
                 reflect=True,
                 **kwargs):
    '''
    fetch tile and augment it
    if rotation and shear is activated, a 3 times larger tile
    is fetched and the final tile is cut out from that after
    rotation/shear.
    '''
    
    #unpack augemtation params and set default values
    
    if augment_params is None:
        rotation_angle = 0
        shear_angle = 0
        flipud = False
        fliplr = False
        rot90 = 0
    else:    
        
        if 'rotation_angle' in augment_params:
            rotation_angle = augment_params['rotation_angle']
        else:
            rotation_angle = 0

        if 'shear_angle' in augment_params:
            shear_angle = augment_params['shear_angle']
        else:
            shear_angle = 0

        if 'flipud' in augment_params:
            flipud = augment_params['flipud']
        else:
            flipud = False

        if 'fliplr' in augment_params:
            fliplr = augment_params['fliplr']
        else:
            fliplr = False

        if 'rot90' in augment_params:
            rot90 = augment_params['rot90']
        else:
            rot90 = 0



    if (size[-2]) == 1 and (size[-1] == 1):
        # if the requested tile is only of size 1 in x and y,
        # augmentation can be omitted, since rotation and flipping always
        # occurs around the center axis.
        return tile_with_reflection(shape, pos, size, get_tile_func,
                                    reflect=reflect, **kwargs)
 

    if (rotation_angle == 0) and (shear_angle == 0):
        res = tile_with_reflection(shape, pos, size, get_tile_func,
                                   reflect=reflect, **kwargs)
        logger.debug(
            'tile_with_reflection dims = {}, {}'.format(res.shape, shape))
        return trafo.flip_image_2d_stack(res, fliplr=fliplr,
                                         flipud=flipud, rot90=rot90)

    
    size = np.array(size)
    pos = np.array(pos)

    size_new = size * 3  # triple tile size if morphing takes place
    pos_new = pos - size
    tile_large = tile_with_reflection(shape, pos_new, size_new, get_tile_func,
                                      reflect=reflect, **kwargs)
    #simple and fast augmentation
    tile_large_flipped = trafo.flip_image_2d_stack(tile_large, fliplr=fliplr,
                                         flipud=flipud, rot90=rot90)
    #slow augmentation
    tile_large_morphed = trafo.warp_image_2d_stack(
        tile_large_flipped, rotation_angle, shear_angle)

    mesh = ut.get_tile_meshgrid(tile_large_morphed.shape, size, size)

    return tile_large_morphed[mesh]


def tile_with_reflection(shape, pos, size, get_tile_func,
                         reflect=True, **kwargs):
    res = inner_tile_size(shape, pos, size)
    pos_transient, size_transient, pos_inside_transient, pad_size = res

    if is_padding(pad_size) and not reflect:
        # if image has to be padded to get the tile
        logger.error('requested tile out of bounds')
        return False

    if is_padding(pad_size) and reflect:
        # if image has to be padded to get the tile and reflection mode is on
        logger.debug('requested tile out of bounds')
        logger.debug('image will be extended with reflection')

    # get transient tile
    transient_tile = get_tile_func(pos=tuple(pos_transient),
                                   size=tuple(size_transient),
                                   **kwargs)
    logger.debug('transient_tile1 dims={}'.format(transient_tile.shape))

    # pad transient tile with reflection
    transient_tile_pad = np.pad(transient_tile, pad_size, mode='symmetric')

    mesh = ut.get_tile_meshgrid(transient_tile_pad.shape,
                                pos_inside_transient, size)

    logger.debug('transient_tile2 dims={}'.format(transient_tile_pad.shape))
    logger.debug('transient_tile3 dims={}'.format(
        transient_tile_pad[mesh].shape))

    return transient_tile_pad[mesh]


def _check_pos_size(pos, size, nr_dim):
    msg = ('Wrong number of image dimensions. '
           'Nr of dimensions MUST be 3 (nr_zslices, nr_x, nr_y), but'
           'is  %s for size and %s for pos') % (str(len(size)), str(len(pos)))

    if (len(pos) != nr_dim) or (len(size) != nr_dim):
        logger.error(msg)
        return False
    return True
