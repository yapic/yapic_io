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
    Currently only works with files from Ilastik v1.2 (storage version 0.1)

    >>> from yapic_io.ilastik_connector import IlastikConnector
    >>> pixel_image_dir = 'yapic_io/test_data/ilastik/pixels_ilastik-multiim-1.2/*.tif'
    >>> ilastik_path = 'yapic_io/test_data/ilastik/ilastik-multiim-1.2.ilp'
    >>> c = IlastikConnector(pixel_image_dir, ilastik_path)
    ... # doctest:+ELLIPSIS
    ...
    >>> print(c)
    IlastikConnector object
    image filepath: yapic_io/test_data/ilastik/pixels_ilastik-multiim-1.2
    label filepath: yapic_io/test_data/ilastik/ilastik-multiim-1.2.ilp
    nr of images: 3
    labelvalue_mapping: [{1: 1, 2: 2}]
    '''
    def handle_lbl_filenames(self, label_filepath):
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
        Overwrites filter_labeled from TiffConnector
        '''
        pairs = [self.filenames[i]for i in range(
            self.image_count()) if self.label_count_for_image(i)]

        tiff_sel = [self.img_path / pair.img for pair in pairs]

        return IlastikConnector(tiff_sel, self.label_path, savepath=self.savepath)

    def split(self, fraction, random_seed=42):
        '''
        Split the images pseudo-randomly into two subsets (both TiffConnectors).
        The first of size `(1-fraction)*N_images`, the other of size `fraction*N_images`
        '''

        img_fnames1, img_fnames2, mask = self._split_img_fnames(
            fraction, random_seed=random_seed)

        conn1 = IlastikConnector(img_fnames1, self.label_path, savepath=self.savepath)
        conn2 = IlastikConnector(img_fnames2, self.label_path, savepath=self.savepath)

        # ensures that both resulting tiff_connectors have the same
        # labelvalue mapping (issue #1)
        conn1.labelvalue_mapping = self.labelvalue_mapping
        conn2.labelvalue_mapping = self.labelvalue_mapping

        return conn1, conn2

    def label_dimensions(self, image_nr):
        '''
        returns dimensions of the label image.
        dims is a 4-element-tuple:

        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)
        '''
        # we do not really know the label dimensions from the *.ilp file
        # so we just use the ones from the image
        return self.image_dimensions(image_nr)

    @lru_cache(maxsize=20)
    def label_tile(self, image_nr, pos_zxy, size_zxy, label_value):
        '''
        returns a 3d zxy boolean matrix where positions of the requested label
        are indicated with True. only mapped labelvalues can be requested.
        '''
        label_filename = str(self.filenames[image_nr].lbl)

        if label_filename is None:
            msg = 'No label matrix file found for image file #{}.'
            logger.warning(msg.format(image_nr))
            return None

        filename, (img, lbl, prediction) = self.ilp[label_filename]
        lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

        C, original_label_value = self._mapped_label_value_to_original(label_value)
        Z, X, Y = pos_zxy
        ZZ, XX, YY = np.array(pos_zxy) + size_zxy
        lbl = lbl[C, Z:ZZ, X:XX, Y:YY]
        lbl = (lbl == original_label_value)

        return lbl


    def check_label_matrix_dimensions(self):
        '''
        Overloads method from tiff connector.
        Method does nothing since it is expected that labelmatrix dimensions
        are correct for Ilastik Projects.
        '''
        return True

    @lru_cache(maxsize=1)
    def original_label_values_for_all_images(self):
        '''
        returns a list of sets. each set corresponds to 1 label channel.
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

            filename, (img, lbl, prediction) = self.ilp[label_filename]
            lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

            C = lbl.shape[0]
            labels = [np.unique(lbl[c,...]) for c in range(C)]
            labels = [set(labels) - {0} for labels in labels]

            labels_per_channel = [l1.union(l2)
                                  for l1, l2 in zip_longest(labels_per_channel, labels, fillvalue=set())]

        return labels_per_channel



    @lru_cache(maxsize = 1500)
    def label_count_for_image(self, image_nr):
        '''
        returns for each label value the number of labels for this image
        '''
        label_filename = str(self.filenames[image_nr].lbl)

        if label_filename is None:
            msg = 'No label matrix file found for image file #{}.'
            logger.warning(msg.format(image_nr))
            return None

        filename, (img, lbl, prediction) = self.ilp[label_filename]
        lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

        C = lbl.shape[0]
        labels = [np.unique(lbl[c,...]) for c in range(C)]

        original_label_count = [{ l: np.count_nonzero(lbl[c,...] == l)
                                  for l in labels[c] if l > 0
                                }
                                for c in range(C)]
        label_count = {self.labelvalue_mapping[c][l]: count
                       for c, orig in enumerate(original_label_count)
                       for l, count in orig.items()}
        return label_count
