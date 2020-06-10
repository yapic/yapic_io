from unittest import TestCase
import os
import numpy as np
from numpy.testing import assert_array_equal
from yapic_io.tiff_connector import TiffConnector
from yapic_io.cellvoy_connector import CellvoyConnector
import logging
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, '../test_data/cellvoyager')


class TestCellvoyConnector(TestCase):

    def test_get_tile_cellvoy(self):

        data_dir = os.path.join(base_path, '../test_data/cellvoyager')
        c = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))

        pxc = c.get_tile(0, (0, 0, 0, 0), (4, 1, 1000, 992))
        assert not np.array_equal(pxc[0, :, :, :], pxc[1, :, :, :])
        assert not np.array_equal(pxc[0, :, :, :], pxc[2, :, :, :])
        assert not np.array_equal(pxc[0, :, :, :], pxc[3, :, :, :])

    def test_compare_cellvoyconnector_with_tiffconnector(self):

        img_files_multi = [os.path.join(
            data_dir,
            '../tif_images/GASPR01S01R01p01E01CD_A05_T0001F001_merge.tif')]

        cc = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        tc = TiffConnector(img_files_multi, 'some/path')

        assert_array_equal(tc.image_dimensions(0), cc.image_dimensions(1))

        pxt = tc.get_tile(0, (3, 0, 0, 0), (1, 1, 1000, 992))
        pxc = cc.get_tile(0, (0, 0, 0, 0), (1, 1, 1000, 992))
        pxc2 = cc.get_tile(1, (0, 0, 0, 0), (1, 1, 1000, 992))
        try:
            assert_array_equal(pxc, pxt)
        except AssertionError:
            assert_array_equal(pxc2, pxt)

        pxt = tc.get_tile(0, (1, 0, 0, 0), (1, 1, 1000, 992))
        pxc = cc.get_tile(0, (1, 0, 0, 0), (1, 1, 1000, 992))
        pxc2 = cc.get_tile(1, (1, 0, 0, 0), (1, 1, 1000, 992))
        try:
            assert_array_equal(pxc, pxt)
        except AssertionError:
            assert_array_equal(pxc2, pxt)

        pxt = tc.get_tile(0, (2, 0, 0, 0), (1, 1, 1000, 992))
        pxc = cc.get_tile(0, (2, 0, 0, 0), (1, 1, 1000, 992))
        pxc2 = cc.get_tile(1, (2, 0, 0, 0), (1, 1, 1000, 992))
        try:
            assert_array_equal(pxc, pxt)
        except AssertionError:
            assert_array_equal(pxc2, pxt)

    def test_label_tile_cellvoy(self):

        cc = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        r = cc.original_label_values_for_all_images()
        assert r == [{1, 2}]
