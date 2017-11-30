import os
import logging
from unittest import TestCase
from yapic_io.ilastik_connector import IlastikConnector
from numpy.testing import assert_array_equal
import numpy as np


logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestIlastikConnector(TestCase):
    def setup_storage_version_12(self):
        img_path = os.path.join(base_path, '../test_data/ilastik')
        lbl_path = os.path.join(base_path, '../test_data/ilastik/ilastik-1.2.ilp')

        return IlastikConnector(img_path, lbl_path)


    def test_dimensions(self):
        c = self.setup_storage_version_12()

        self.assertEqual(c.image_dimensions(0), c.label_dimensions(0))


    def test_tiles(self):
        c = self.setup_storage_version_12()

        lbl_value = 2
        pos_czxy = (0, 0, 0, 0)
        size_czxy = (1, 1, 1, 1)

        img_tile = c.get_tile(0, pos_czxy, size_czxy)
        lbl_tile = c.label_tile(0, pos_czxy[1:], size_czxy[1:], lbl_value)

        self.assertEqual(img_tile.shape[1:], lbl_tile.shape)


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
        expected_counts = { 1: 1, 2: 1, 3:1 }

        self.assertEqual(actual_counts, expected_counts)


    def test_constructor(self):
        img_path = os.path.join(base_path, '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        lbl_path = os.path.join(base_path, '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, lbl_path)
        print(c.filenames)
        assert False






