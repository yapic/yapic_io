import os
import logging
from unittest import TestCase
from yapic_io.ilastik_connector import IlastikConnector
from numpy.testing import assert_array_equal
import numpy as np
from pprint import pprint
from pathlib import Path

logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)


class TestIlastikConnector(TestCase):
    def setup_storage_version_12(self):
        img_path = os.path.join(base_path, '../test_data/ilastik')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-1.2.ilp')

        return IlastikConnector(img_path, lbl_path)

    def test_dimensions(self):
        c = self.setup_storage_version_12()

        assert_array_equal(c.image_dimensions(0), c.label_dimensions(0))

    def test_tiles(self):
        c = self.setup_storage_version_12()

        lbl_value = 2
        pos_czxy = (0, 0, 0, 0)
        size_czxy = (1, 1, 1, 1)

        img_tile = c.get_tile(0, pos_czxy, size_czxy)
        lbl_tile = c.label_tile(0, pos_czxy[1:], size_czxy[1:], lbl_value)

        assert_array_equal(img_tile.shape[1:], lbl_tile.shape)

    def test_label_tiles(self):
        c = self.setup_storage_version_12()

        lbl_value = 1
        pos_czxy = (0, 4, 0, 0)
        size_czxy = (1, 1, 2, 4)
        val = np.array([[[False, False, False, False],
                         [False, False, False, True]]])
        lbl_tile = c.label_tile(0, pos_czxy[1:], size_czxy[1:], lbl_value)
        assert_array_equal(lbl_tile, val)

        lbl_value = 2
        pos_czxy = (0, 1, 0, 0)
        size_czxy = (1, 1, 2, 4)
        val = np.array([[[False, False, False, False],
                         [False,  True, False, False]]])
        lbl_tile = c.label_tile(0, pos_czxy[1:], size_czxy[1:], lbl_value)
        assert_array_equal(lbl_tile, val)

        lbl_value = 3
        pos_czxy = (0, 7, 0, 0)
        size_czxy = (1, 1, 2, 4)
        val = np.array([[[False, False, False, False],
                         [False,  True, False, False]]])
        lbl_tile = c.label_tile(0, pos_czxy[1:], size_czxy[1:], lbl_value)
        assert_array_equal(lbl_tile, val)

    def test_label_count(self):
        c = self.setup_storage_version_12()

        actual_counts = c.label_count_for_image(0)
        expected_counts = {1: 1, 2: 1, 3: 1}

        self.assertEqual(actual_counts, expected_counts)

        # print(c.load_label_matrix(0).shape)
        #assert False

    def test_constructor(self):
        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, lbl_path)

        lbl_identifiers = [Path('pixels_ilastik-multiim-1.2/20width_23height_3slices_2channels.tif'),
                           Path('pixels_ilastik-multiim-1.2/34width_28height_2slices_2channels.tif'),
                           Path('pixels_ilastik-multiim-1.2/6width_4height_3slices_2channels.tif')]

        assert_array_equal(lbl_identifiers, [lbl for im, lbl in c.filenames])

    def test_constructor_with_subset(self):

        # by passing a list of tiff filenames to IlastikConnector (rather than a wildcard)
        # an image subset of the ilastik project can be selected

        tiff_dir = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        selected_tiffs = [os.path.join(tiff_dir, '20width_23height_3slices_2channels.tif'),
                          os.path.join(tiff_dir, '6width_4height_3slices_2channels.tif')]

        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(selected_tiffs, lbl_path)

        lbl_identifiers = [Path('pixels_ilastik-multiim-1.2/20width_23height_3slices_2channels.tif'),
                           Path('pixels_ilastik-multiim-1.2/6width_4height_3slices_2channels.tif')]

        pprint(c.filenames)
        assert_array_equal(lbl_identifiers, [lbl for im, lbl in c.filenames])

    def test_label_tile(self):
        import warnings
        warnings.warn(('test_label_tile() should be reimplemented when '
                       'IlastikConnector is fixes!'), FutureWarning)
        '''
        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, lbl_path)

        mat_val = np.array([[0.,  0.,  0.,  0.,  0.,  0.,  0.],
                            [0.,  0.,  2.,  2.,  2.,  0.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  2.,  0.],
                            [0.,  2.,  2.,  2.,  2.,  0.,  0.],
                            [0.,  0.,  0.,  2.,  0.,  0.,  0.],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        lbl = c.label_tile(0, (0,0,0), (1,19,17), 2)
        assert_array_equal(lbl[0, 6:18, 9:16], mat_val != 0)

        mat_val = np.array([[0.,  0.,  0.,  1.,  0.,  0.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  1.,  0.],
                            [0.,  1.,  1.,  1.,  1.,  0.,  0.],
                            [0.,  0.,  0.,  1.,  0.,  0.,  0.],
                            [0.,  0.,  0.,  0.,  0.,  0.,  0.]])

        lbl = c.label_tile(0, (0,0,0), (1,14,9), 1)
        assert_array_equal(lbl[0, :13, 1:8], mat_val != 0)
    '''

    def test_filter_labeled(self):

        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')

        c = IlastikConnector(img_path, lbl_path)
        c_filtered = c.filter_labeled()

        labelnames = [Path('pixels_ilastik-multiim-1.2/20width_23height_3slices_2channels.tif'),
                      Path('pixels_ilastik-multiim-1.2/34width_28height_2slices_2channels.tif'),
                      Path('pixels_ilastik-multiim-1.2/6width_4height_3slices_2channels.tif')]

        labelnames_flt = [Path('pixels_ilastik-multiim-1.2/20width_23height_3slices_2channels.tif'),
                          Path('pixels_ilastik-multiim-1.2/34width_28height_2slices_2channels.tif')]

        assert_array_equal(labelnames, [lbl for im, lbl in c.filenames])
        assert_array_equal(labelnames_flt, [
                         lbl for im, lbl in c_filtered.filenames])

    def test_split(self):
        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')

        c = IlastikConnector(img_path, lbl_path)

        c1, c2 = c.split(0.3)

        assert_array_equal(c1.image_count() + c2.image_count(), c.image_count())
