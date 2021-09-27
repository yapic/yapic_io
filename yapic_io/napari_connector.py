from yapic_io.tiff_connector import TiffConnector
from pathlib import Path
import numpy as np
import collections
import logging
import os
from itertools import zip_longest
from functools import lru_cache
import sparse
import typing
import h5py


FilePair = collections.namedtuple('FilePair', ['img', 'lbl'])
logger = logging.getLogger(os.path.basename(__file__))


def reconstruct_layer(layer_array: np.array, shape: tuple) -> np.array:
    """Returns a numpy array corresponding to the data reconstruction from a
    sparse version (COO list) of it.

        Parameters
        ----------
        layer_array: Numpy array
            Sparse array version (COO list) of the layer data
        shape: tuple
            shape of the original layer data array

        Returns
        -------
        np.array
            Full version of the layer data
    """
    coords = layer_array[:-1]
    values = layer_array[-1]
    tmp_sparse = sparse.COO(coords=coords, data=values, shape=shape)
    return tmp_sparse.todense()


class NapariConnector(TiffConnector):
    def __init__(self, img_filepath, label_filepath, savepath=None):
        # Dictionary of list telling labeled slices (non-zero matrices)
        self.labeled_slices = dict()
        super().__init__(img_filepath, label_filepath, savepath=savepath)

    def _assemble_filenames(self, pairs):
        self.filenames = [FilePair(Path(img), lbl)
                          for img, lbl in pairs if lbl]
        print('filenames in napariconnector')
        print(self.filenames)

    def _handle_lbl_filenames(self, label_filepath):
        self.h5 = NapariStorage(h5_path=label_filepath, max_dim=4)
        lbl_filenames = self.h5.get_labels_names()
        return label_filepath, lbl_filenames

    def __repr__(self):
        infostring = \
            'NapariConnector object\n' \
            'image filepath: {}\n' \
            'label filepath: {}\n'\
            'nr of images: {}\n'\
            'labelvalue_mapping: {}'.format(self.img_path,
                                            self.label_path,
                                            self.image_count(),
                                            self.labelvalue_mapping)
        return infostring

    def _new_label(self, label_value):

        new_list = []
        new_list = [x for x in label_value[1] if x[1] is not None]

        for x in label_value:
            if label_value[1] is not None:
                new_list.append(x)
            else:
                pass
        label_value = new_list
        return label_value

    def effective_slices(self):
        return self.labeled_slices

    def filter_labeled(self):
        '''
        Removes images without labels.

        Returns
        -------
        NapariConnector
            Connector object containing only images with labels.
        '''
        pairs = [self.filenames[i]for i in range(
            self.image_count()) if self.label_count_for_image(i)]
        tiff_sel = [self.img_path / pair.img for pair in pairs]

        return NapariConnector(tiff_sel, self.label_path,
                               savepath=self.savepath)

    def split(self, fraction, random_seed=42):
        '''
        Split the images pseudo-randomly into two Connector subsets.

        The first of size `(1-fraction)*N_images`, the other of size
        `fraction*N_images`

        Parameters
        ----------
        fraction : float
        random_seed : float, optional

        Returns
        -------
        connector_1, connector_2
        '''

        img_fnames1, img_fnames2, mask = self._split_img_fnames(
            fraction, random_seed=random_seed)

        conn1 = NapariConnector(img_fnames1, self.label_path,
                                savepath=self.savepath)
        conn2 = NapariConnector(img_fnames2, self.label_path,
                                savepath=self.savepath)

        # ensures that both resulting connectors have the same
        # labelvalue mapping (issue #1)
        conn1.labelvalue_mapping = self.labelvalue_mapping
        conn2.labelvalue_mapping = self.labelvalue_mapping

        return conn1, conn2

    def label_tile(self, image_nr, pos_zxy, size_zxy, label_value):
        Z, X, Y = pos_zxy
        ZZ, XX, YY = np.array(pos_zxy) + size_zxy
        _, original_label_value = self._mapped_label_value_to_original(
            label_value)

        slices = self._open_label_file(image_nr)
        if slices is None:
            # return tile with False values
            return np.zeros(size_zxy) != 0
        tile = slices[Z: ZZ, Y: YY, X: XX, 0]
        tile = np.moveaxis(tile, (0, 1, 2), (0, 2, 1))

        tile = (tile == original_label_value)
        return tile

    def _open_label_file(self, image_nr):
        label_filename = self.filenames[image_nr].lbl

        if label_filename is None:
            logger.warning(
                'no label matrix file found for image file %s', str(image_nr))
            return None

        logger.debug('Trying to load labelmat {} in {} Napari project'.format(
            label_filename, self.label_path))

        label_data = self.h5.get_array_data('labels', label_filename)
        label_n_dim = self.h5.n_dims('labels', label_filename)

        if label_n_dim == 2:
            label_data = np.expand_dims(label_data, axis=-1)  # channel
            label_data = np.expand_dims(label_data, axis=0)  # z dim
        elif label_n_dim == 3:
            label_data = np.expand_dims(label_data, axis=-1)  # channel

        if image_nr not in self.labeled_slices.keys():
            self.labeled_slices[image_nr] = self.h5.filled_slices(
                'labels', label_filename)

        return label_data

    def label_matrix_dimensions(self, image_nr):
        '''
        Get dimensions of the label image.


        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        (nr_channels, nr_zslices, nr_x, nr_y)
            Labelmatrix shape.
        '''
        lbl = self._open_label_file(image_nr)
        if lbl is None:
            return

        output = list(lbl.shape)
        ch = output.pop()
        output[-2], output[-1] = output[-1], output[-2]
        output.insert(0, ch)

        return output

    @lru_cache(maxsize=1)
    def original_label_values_for_all_images(self):
        '''
        Get all unique label values per image.

        Returns
        -------
        list
            List of sets. Each set corresponds to 1 label channel.
            each set contains the label values of that channel.
            E.g. `[{91, 109, 150}, {90, 100}]` for two label channels
        '''
        labels_per_channel = []

        for image_nr in range(self.image_count()):
            label_filename = str(self.filenames[image_nr].lbl)

            if label_filename is None:
                msg = 'No label matrix file found for image file #{}.'
                logger.warning(msg.format(image_nr))
                return None
            print('label filename')
            print(label_filename)
            lbl = self._open_label_file(image_nr)

            lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

            C = lbl.shape[0]
            labels = [np.unique(lbl[c, ...]) for c in range(C)]
            labels = [set(labels) - {0} for labels in labels]

            labels_per_channel = [l1.union(l2)
                                  for l1, l2 in zip_longest(labels_per_channel,
                                                            labels,
                                                            fillvalue=set())]

        return labels_per_channel

    @lru_cache(maxsize=1500)
    def label_count_for_image(self, image_nr):
        '''
        Get number of labels per labelvalue for an image.

        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        dict
        '''
        label_filename = str(self.filenames[image_nr].lbl)

        if label_filename is None:
            msg = 'No label matrix file found for image file #{}.'
            logger.warning(msg.format(image_nr))
            return None

        lbl = self._open_label_file(image_nr)

        lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

        C = lbl.shape[0]
        labels = [np.unique(lbl[c, ...]) for c in range(C)]

        original_label_count = [{l: np.count_nonzero(lbl[c, ...] == l)
                                 for l in labels[c] if l > 0}
                                for c in range(C)]
        label_count = {self.labelvalue_mapping[c][l]: count
                       for c, orig in enumerate(original_label_count)
                       for l, count in orig.items()}
        return label_count


class NapariStorage():
    def __init__(self, h5_path, max_dim=np.inf):
        self.f = h5py.File(h5_path, 'r')
        self.max_dim = max_dim

    def __iter__(self):
        '''
        Returns (filename, data, type) of the napari layer
        '''
        for layer_type in self.f:  # check folder of Napari layer types
            # only images and labels are required in YAPiC
            if layer_type in ['image', 'labels']:
                # iterate over specific layer
                for layer_name in self.f[layer_type]:
                    if self.dim_check(layer_type, layer_name):
                        layer_data = self.get_array_data(
                            layer_type, layer_name)
                        yield layer_name, layer_data, layer_type

    def get_array_data(self, layer_type: str, layer_name: str) -> np.array:
        '''
        Returns array data of Napari layer. it considers only image and label
        Napari layers (those supported by YAPiC)
            the array dimensions are (?, z, y, x, c)
        '''
        try:
            assert layer_type in ['image', 'labels']
        except AssertionError:
            raise ValueError(
                'Supported Napari layers are image and labels only')
        else:
            try:
                assert layer_name in self.f[layer_type]
            except AssertionError:
                raise ValueError(
                    'There is no Napari layer with the name {}'.format(
                        layer_name))

        napari_layer = self.f[layer_type][layer_name]
        if layer_type == 'labels':
            original_shape = tuple(napari_layer.attrs['shape'])
            if napari_layer.attrs['is_sparse']:
                array_data = reconstruct_layer(np.array(napari_layer),
                                               original_shape)
            else:
                array_data = napari_layer[:]
        else:
            array_data = np.array(napari_layer)
        return array_data
    # This function might be changed in case the hdf5
    # compression protocol is changed! 
    def filled_slices(self, layer_type: str, layer_name: str) -> list:
        '''
        Returns a list of indices specifying which slices have labels
        (values different than 0)
        '''
        napari_layer = self.f[layer_type][layer_name]
        data = np.array(napari_layer)
        if layer_type == 'labels':
            if napari_layer.attrs['is_sparse']:
                # check if the sparse label includes z-dim
                if napari_layer.shape[0] > 3:
                    return list(np.unique(data[0, :]))
            else:
                if len(data.shape) > 2:
                    return [z_slice for z_slice in range(data.shape[0]) 
                            if len(np.unique(data[0, :])) > 1]
        return [0]  # when the images are 2D there is only one slice

    def excluded_layers(self) -> dict:
        '''
        Returns a dictionary of excluded layers due to the maximum dimension
        and not used layer types.

        keys = Napari layer types
        values = Set of skipped layers
        '''
        skipped_types = set(self.f.keys()) - {'image', 'labels'}
        output_dict = {layer_type: set(
            self.f[layer_type].keys()) for layer_type in skipped_types}
        for layer_type in ['image', 'labels']:
            tmp_names = [name for name in self.f[layer_type].keys(
            ) if not self.dim_check(layer_type, name)]
            if len(tmp_names) > 0:
                output_dict[layer_type] = set(tmp_names)
        return output_dict

    def n_dims(self, layer_type: str, layer_name: str) -> int:
        '''
        Returns the number of dimensions of Napari layer
        '''
        napari_layer = self.f[layer_type][layer_name]
        if layer_type == 'labels' and napari_layer.attrs['is_sparse']:
            shape = napari_layer.attrs['shape']
        else:
            shape = napari_layer.shape
        return len(shape)

    def dim_check(self, layer_type: str, layer_name: str) -> bool:
        return bool(self.n_dims(layer_type, layer_name) <= self.max_dim)

    def get_labels_names(self) -> list:
        '''
        Returns a list of all available Napari label layers considering
        the maximum dimension.
        '''
        return [lbl_name for lbl_name in self.f['labels'].keys()
                if self.dim_check('labels', lbl_name)]

    def get_image_names(self) -> list:
        '''
        Returns a list of all available Napari image layers considering
        the maximum dimension.
        '''
        return [im_name for im_name in self.f['image'].keys()
                if self.dim_check('image', im_name)]

    def __len__(self):
        '''
        Returns the number of image and labels (Those supported by YAPiC)
        in the Napari project considering the maximum dimension.
        '''
        return self.number_of_labels() + self.number_of_images()

    def number_of_labels(self) -> int:
        '''
        Returns the number of label layers in the Napari project
        considering the maximum dimension.
        '''
        return len(self.get_labels_names())

    def number_of_images(self) -> int:
        '''
        Returns the number of image layers in the Napari project
        considering the maximum dimension.
        '''
        return len(self.get_image_names())
