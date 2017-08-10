import os
import logging
import glob
from functools import lru_cache

import numpy as np

logger = logging.getLogger(os.path.basename(__file__))

import pyilastik

from yapic_io.tiff_connector import TiffConnector
from yapic_io import utils


class IlastikConnector(TiffConnector):
    '''
    Currently only works with files from Ilastik v1.2 (storage version 0.1)
    '''
    def __init__(self, img_filepath, label_filepath, *args, **kwds):
        self.ilp = pyilastik.read_project(label_filepath, skip_image=True)

        if not img_filepath.endswith('.tif'):
            img_filepath = os.path.join(img_filepath, '*.tif')

        img_path_list = glob.glob(img_filepath)

        # map from (potentially) paths from another machine to current machine
        lbl_path_list = self.ilp.image_path_list()
        lbl_path_list = [ img_path for lbl_path, img_path in zip(lbl_path_list, img_path_list)
                          if os.path.basename(lbl_path) == os.path.basename(img_path)]

        super().__init__(img_path_list, lbl_path_list, *args, **kwds)


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


    @lru_cache(maxsize = 20)
    def load_label_matrix(self, image_nr, original_labelvalues=False):
        '''
        returns a 4d labelmatrix with dimensions czxy.
        the albelmatrix consists of zeros (no label) or the respective
        label value.

        if original_labelvalues is False, the mapped label values are returned,
        otherwise the original labelvalues.
        '''
        label_filename = self.filenames[image_nr].lbl
        img_shape = self.image_dimensions(image_nr)

        if label_filename is None:
            msg = 'no label matrix file found for image file #{}'.format(image_nr)
            logger.warning(msg)
            return None

        filename, (img, lbl, prediction) = self.ilp[label_filename]
        lbl = np.transpose(lbl, (3, 0, 2, 1))

        # acutal label size is unknown in ilastik project file, so we pad with zeros
        pad_width = [(0, i - l) for i, l in zip(img_shape, lbl.shape)]
        lbl = np.pad(lbl, pad_width, 'constant', constant_values=(0,0))

        if original_labelvalues:
            return lbl

        return utils.assign_slice_by_slice(self.labelvalue_mapping, lbl)

