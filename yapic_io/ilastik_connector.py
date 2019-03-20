from itertools import zip_longest
import os
import logging
from functools import lru_cache
import numpy as np
import pyilastik
from yapic_io.tiff_connector import TiffConnector

logger = logging.getLogger(os.path.basename(__file__))


class IlastikConnector(TiffConnector):
    '''
    Implementation of Connector for tiff images up to 4 dimensions and
    corresponding Ilastik_ project file. The Ilastik_ Project file
    is supposed to contain manually drawn labels for all tiff files specified
    with img_filepath

    .. _Ilastik: http://www.ilastik.org/

    Parameters
    ----------
    img_filepath : str or list of str
        Path to source pixel images (use wildcards for filtering)
        or a list of filenames.
    label_filepath : str
        Path to one Ilastik Project File (with extension ilp). Ilastik_
        versions from 1.3 on are supported.
    savepath : str, optional
        Directory to save pixel classifiaction results as probability
        images.

    Notes
    -----
    Label images and pixel images have to be equal in zxy dimensions,
    but can differ in nr of channels.

    Labels can be read from multichannel images. This is needed for
    networks with multiple output layers. Each channel is assigned one
    output layer. Different labels from different channels can overlap
    (can share identical xyz positions).

    Files from Ilastik v1.2 and v1.3 are supported (storage version 0.1).

    Examples
    --------
    >>> from yapic_io.ilastik_connector import IlastikConnector
    >>> img_dir = 'yapic_io/test_data/ilastik/pixels_ilastik-multiim-1.2/*.tif'
    >>> ilastik_path = 'yapic_io/test_data/ilastik/ilastik-multiim-1.2.ilp'
    >>> c = IlastikConnector(img_dir, ilastik_path)
    ... # doctest:+ELLIPSIS
    ...
    >>> print(c)
    IlastikConnector object
    image filepath: yapic_io/test_data/ilastik/pixels_ilastik-multiim-1.2
    label filepath: yapic_io/test_data/ilastik/ilastik-multiim-1.2.ilp
    nr of images: 3
    labelvalue_mapping: [{1: 1, 2: 2}]

    See Also
    --------
    yapic_io.tiff_connector.TiffConnector
    '''
    def _handle_lbl_filenames(self, label_filepath):
        label_path = label_filepath
        self.ilp = pyilastik.read_project(label_filepath, skip_image=True)
        lbl_filenames = self.ilp.image_path_list()

        return label_path, lbl_filenames

    def __repr__(self):
        infostring = \
            'IlastikConnector object\n' \
            'image filepath: {}\n' \
            'label filepath: {}\n'\
            'nr of images: {}\n'\
            'labelvalue_mapping: {}'.format(self.img_path,
                                            self.label_path,
                                            self.image_count(),
                                            self.labelvalue_mapping)
        return infostring

    def filter_labeled(self):
        '''
        Removes images without labels.

        Returns
        -------
        IlastikConnector
            Connector object containing only images with labels.
        '''
        pairs = [self.filenames[i]for i in range(
            self.image_count()) if self.label_count_for_image(i)]

        tiff_sel = [self.img_path / pair.img for pair in pairs]

        return IlastikConnector(tiff_sel, self.label_path,
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

        conn1 = IlastikConnector(img_fnames1, self.label_path,
                                 savepath=self.savepath)
        conn2 = IlastikConnector(img_fnames2, self.label_path,
                                 savepath=self.savepath)

        # ensures that both resulting connectors have the same
        # labelvalue mapping (issue #1)
        conn1.labelvalue_mapping = self.labelvalue_mapping
        conn2.labelvalue_mapping = self.labelvalue_mapping

        return conn1, conn2

    @lru_cache(maxsize=20)
    def label_tile(self, image_nr, pos_zxy, size_zxy, label_value):
        '''
        Get 3d zxy boolean matrix where positions of the requested label
        are indicated with True. Only mapped labelvalues can be requested.

        dimension order: (z, x, y)

        Parameters
        ----------
        image_nr : int
            Index of image.
        pos_zxy : (zslice, x, y)
            Upper left position of subsection.
        label_value : int
            Id of the label.

        Returns
        -------
        numpy.ndarray
            3D subsection of labelmatrix as boolean mask in dimension order
            (z, x, y)
        '''

        slices = np.array([[pos_zxy[0], pos_zxy[0] + size_zxy[0]],  # z
                           [pos_zxy[2], pos_zxy[2] + size_zxy[2]],  # y
                           [pos_zxy[1], pos_zxy[1] + size_zxy[1]],  # x
                           [0, 1]])  # c

        if self.ilp.n_dims(image_nr) == 0:  # no labels in image
            return np.zeros(size_zxy) > 0

        elif self.ilp.n_dims(image_nr) == 4:  # z-stacks
            lbl = self.ilp.tile(image_nr, slices)

        elif self.ilp.n_dims(image_nr) == 3:  # 2d images
            lbl = self.ilp.tile(image_nr, slices[1:, :])
            lbl = np.expand_dims(lbl, axis=0)  # add z axis

        # zyxc to czxy
        lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

        C, original_label_value = self._mapped_label_value_to_original(
                                         label_value)
        lbl = (lbl == original_label_value)

        return lbl[0, :, :, :]

    def check_label_matrix_dimensions(self):
        '''
        Notes
        -----
        Overloads method from tiff connector.
        Method does nothing since it is expected that labelmatrix dimensions
        are correct for Ilastik Projects.
        '''
        return True

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

            _, (img, lbl, _) = self.ilp[label_filename]
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

        _, (img, lbl, _) = self.ilp[label_filename]
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
