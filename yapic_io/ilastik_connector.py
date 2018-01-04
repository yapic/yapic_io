import os
import logging
import glob
from functools import lru_cache

import numpy as np

logger = logging.getLogger(os.path.basename(__file__))

import pyilastik

from yapic_io.tiff_connector import TiffConnector, handle_img_filenames, FilePair
from yapic_io import utils as ut


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

        tiff_sel = [os.path.join(self.img_path, pair.img) for pair in pairs]

        return IlastikConnector(tiff_sel, self.label_path,
                                savepath=self.savepath,
                                multichannel_pixel_image=self.multichannel_pixel_image,
                                zstack=self.zstack)

    def split(self, fraction, random_seed=42):
        '''
        Split the images pseudo-randomly into two subsets (both TiffConnectors).
        The first of size `(1-fraction)*N_images`, the other of size `fraction*N_images`
        '''

        img_fnames1, img_fnames2, mask = self._split_img_fnames(
            fraction, random_seed=random_seed)

        conn1 = IlastikConnector(img_fnames1, self.label_path,
                                 savepath=self.savepath,
                                 multichannel_pixel_image=self.multichannel_pixel_image,
                                 zstack=self.zstack)
        conn2 = IlastikConnector(img_fnames2, self.label_path,
                                 savepath=self.savepath,
                                 multichannel_pixel_image=self.multichannel_pixel_image,
                                 zstack=self.zstack)

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
    def load_label_matrix(self, image_nr, original_labelvalues=False):
        '''
        returns a 4d labelmatrix with dimensions czxy.
        the albelmatrix consists of zeros (no label) or the respective
        label value.

        if original_labelvalues is False, the mapped label values are returned,
        otherwise the original labelvalues.
        '''
        label_filename = str(self.filenames[image_nr].lbl)
        img_shape = self.image_dimensions(image_nr)

        if label_filename is None:
            msg = 'no label matrix file found for image file #{}'.format(
                image_nr)
            logger.warning(msg)
            return None

        filename, (img, lbl, prediction) = self.ilp[label_filename]

        lbl = np.transpose(lbl, (3, 0, 2, 1)).astype(int)

        if lbl.size == 0:
            lbl = np.zeros((1, 1, 1, 0))

        # acutal label size is unknown in ilastik project file, so we pad with zeros
        #pad_width = [(0, i - l) for i, l in zip(img_shape, lbl.shape)]
        # padding for label channel is always zero (always one label channel
        # for ilastik data)
        pad_width = [(0, 0)] + [(0, i - l)
                                for i, l in zip(img_shape[1:], lbl.shape[1:])]
        lbl = np.pad(lbl, pad_width, 'constant', constant_values=(0, 0))

        if original_labelvalues:
            return lbl

        return ut.assign_slice_by_slice(self.labelvalue_mapping, lbl)

    def check_label_matrix_dimensions(self):
        '''
        Overloads method from tiff connector.
        Method does nothing since it is expected that labelmatrix dimensions
        are correct for Ilastik Projects.
        '''
        return True
