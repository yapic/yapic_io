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

    def test_constructor(self):
        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, lbl_path)

        lbl_identifiers = \
            [Path(('pixels_ilastik-multiim-1.2/'
                   '20width_23height_3slices_2channels.tif')),
             Path(('pixels_ilastik-multiim-1.2/'
                   '34width_28height_2slices_2channels.tif')),
             Path(('pixels_ilastik-multiim-1.2/'
                   '6width_4height_3slices_2channels.tif'))]

        assert_array_equal(lbl_identifiers, [lbl for im, lbl in c.filenames])

    def test_constructor_with_subset(self):

        # by passing a list of tiff filenames to IlastikConnector
        # (rather than a wildcard) an image subset of the ilastik project
        # can be selected

        tiff_dir = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        selected_tiffs = [os.path.join(
                          tiff_dir,
                          '20width_23height_3slices_2channels.tif'),
                          os.path.join(
                          tiff_dir,
                          '6width_4height_3slices_2channels.tif')]

        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(selected_tiffs, lbl_path)

        lbl_identifiers = \
            [Path(('pixels_ilastik-multiim-1.2/'
                   '20width_23height_3slices_2channels.tif')),
             Path(('pixels_ilastik-multiim-1.2/'
                   '6width_4height_3slices_2channels.tif'))]

        pprint(c.filenames)
        assert_array_equal(lbl_identifiers, [lbl for im, lbl in c.filenames])

    def test_label_tile(self):

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

        lbl = c.label_tile(0, (0, 0, 0), (1, 19, 17), 2)
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

        lbl = c.label_tile(0, (0, 0, 0), (1, 14, 9), 1)
        assert_array_equal(lbl[0, :13, 1:8], mat_val != 0)

    def test_labels_for_ilastik_versions_12_133_are_equal(self):

        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.3.3.ilp')
        c13 = IlastikConnector(img_path, lbl_path)

        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c12 = IlastikConnector(img_path, lbl_path)

        lbl12 = c12.label_tile(0, (0, 0, 0), (1, 19, 17), 2)
        lbl13 = c13.label_tile(0, (0, 0, 0), (1, 19, 17), 2)
        assert_array_equal(lbl12, lbl13)

        assert c12.label_count_for_image(0) == c13.label_count_for_image(0)

    def test_label_tile_purkinjedata(self):

        p = os.path.join(base_path, '../test_data/ilastik/purkinjetest')
        img_path = os.path.join(p, 'images')
        lbl_path = os.path.join(p, 'ilastik-1.2.2post1mac.ilp')

        c = IlastikConnector(img_path, lbl_path)
        print(c.filenames)
        print(c.image_count)

        image_id = 0  # 769_cerebellum_5M41_subset_1.tif
        pos_zxy = (0, 309, 212)
        size_zxy = (1, 4, 5)

        val = np.array([[[True, False, False, False, False],
                         [True, True, False, False, False],
                         [True, True, True, False, False],
                         [True, True, True, False, False]]])
        lbl = c.label_tile(image_id, pos_zxy, size_zxy, 2)
        assert_array_equal(lbl, val)

        val = np.array([[[False, False, False, False, True],
                         [False, False, False, False, True],
                         [False, False, False, True, True],
                         [False, False, False, True, True]]])
        lbl = c.label_tile(image_id, pos_zxy, size_zxy, 4)
        assert_array_equal(lbl, val)

    def test_labeltile_dimensions_purkinjedata(self):

        p = os.path.join(base_path, '../test_data/ilastik/purkinjetest')
        img_path = os.path.join(p, 'images')
        lbl_path = os.path.join(p, 'ilastik-1.2.2post1mac.ilp')

        c = IlastikConnector(img_path, lbl_path)

        image_id = 3  # 769_cerebellum_5M41_subset_1.tif
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 1047, 684)  # whole image

        lbl = c.label_tile(image_id, pos_zxy, size_zxy, 4)
        self.assertEqual(lbl.shape, size_zxy)

    def test_labeltile_for_image_without_labels(self):
        p = os.path.join(base_path, '../test_data/ilastik/purkinjetest')
        img_path = os.path.join(p, 'images')
        lbl_path = os.path.join(p, 'ilastik-1.2.2post1mac.ilp')

        c = IlastikConnector(img_path, lbl_path)
        print(c.filenames)
        print(c.image_count)

        image_id = 2  # 769_cerebellum_5M41_subset_1.tif
        pos_zxy = (0, 309, 212)
        size_zxy = (1, 4, 5)

        val = np.array([[[False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False],
                         [False, False, False, False, False]]])

        lbl = c.label_tile(image_id, pos_zxy, size_zxy, 3)
        assert_array_equal(lbl, val)

    def test_multi_channel_multi_z(self):

        p = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(p, 'images')
        lbl_path = os.path.join(p, 'x15_y10_z2_c4_classes2.ilp')
        c = IlastikConnector(img_path, lbl_path)
        pos_zxy = (0, 0, 0)
        size_zxy = (2, 15, 10)

        lbl = c.label_tile(0, pos_zxy, size_zxy, 1)
        lbl_pos = [[0, 2, 1], [0, 2, 1], [0, 8, 6], [0, 9, 6], [1, 4, 3]]
        [self.assertTrue(lbl[pos[0], pos[1], pos[2]]) for pos in lbl_pos]

        lbl = c.label_tile(0, pos_zxy, size_zxy, 2)
        lbl_pos = [[0, 2, 2], [0, 3, 2], [0, 8, 7], [0, 9, 7]]
        [self.assertTrue(lbl[pos[0], pos[1], pos[2]]) for pos in lbl_pos]

        self.assertFalse(lbl[0, 0, 0])


        p = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(p, 'images')
        lbl_path = os.path.join(p, 'x15_y10_z2_c4_classes2_ilastik1.3.3.ilp')
        c = IlastikConnector(img_path, lbl_path)
        pos_zxy = (0, 0, 0)
        size_zxy = (2, 15, 10)

        lbl = c.label_tile(0, pos_zxy, size_zxy, 1)
        lbl_pos = [[0, 2, 1], [0, 2, 1], [0, 8, 6], [0, 9, 6], [1, 4, 3]]
        [self.assertTrue(lbl[pos[0], pos[1], pos[2]]) for pos in lbl_pos]

        lbl = c.label_tile(0, pos_zxy, size_zxy, 2)
        lbl_pos = [[0, 2, 2], [0, 3, 2], [0, 8, 7], [0, 9, 7]]
        [self.assertTrue(lbl[pos[0], pos[1], pos[2]]) for pos in lbl_pos]

        self.assertFalse(lbl[0, 0, 0])

    def test_filter_labeled(self):

        img_path = os.path.join(
            base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')

        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')

        c = IlastikConnector(img_path, lbl_path)
        c_filtered = c.filter_labeled()

        labelnames = [Path(('pixels_ilastik-multiim-1.2/'
                            '20width_23height_3slices_2channels.tif')),
                      Path(('pixels_ilastik-multiim-1.2/'
                            '34width_28height_2slices_2channels.tif')),
                      Path(('pixels_ilastik-multiim-1.2/'
                            '6width_4height_3slices_2channels.tif'))]

        labelnames_flt = [Path(('pixels_ilastik-multiim-1.2/'
                                '20width_23height_3slices_2channels.tif')),
                          Path(('pixels_ilastik-multiim-1.2/'
                                '34width_28height_2slices_2channels.tif'))]

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

        assert_array_equal(c1.image_count() + c2.image_count(),
                           c.image_count())
