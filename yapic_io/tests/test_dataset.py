from unittest import TestCase
import os
import numpy as np
from yapic_io.tiff_connector import TiffConnector
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.cellvoy_connector import CellvoyConnector
from yapic_io.connector import io_connector
from yapic_io.dataset import Dataset
from yapic_io.utils import get_tile_meshgrid
import yapic_io.dataset as ds
from pprint import pprint
import logging
from numpy.testing import assert_array_equal, assert_array_almost_equal
import sys
import pytest
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.WARNING)
base_path = os.path.dirname(__file__)


def c(*a):
    return np.array(a)


class TestDataset(TestCase):

    def test_pixel_statistics(self):

        data_dir = os.path.join(base_path, '../test_data/cellvoyager')
        c = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        d = Dataset(c)
        channels = [0, 1, 2, 3]

        stats = d.pixel_statistics([0, 1, 2, 3], n_tiles=1000)

        assert len(stats) == len(channels)
        for stat in stats:
            assert len(stat) == 2
            assert stat[0] < stat[1]

    def test_n_images(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path, 'path/to/nowhere/')
        d = Dataset(c)

        np.testing.assert_array_equal(d.n_images, 3)

    def test_get_padding_size_1(self):

        shape = c(7, 11)
        pos = c(5, 5)
        size = c(3, 5)
        res = ds.inner_tile_size(shape, pos, size)
        np.testing.assert_array_equal([(0, 1), (0, 0)], res[-1])

    def test_get_padding_size_2(self):

        shape = c(7, 11)
        pos = c(-3, 0)
        size = c(3, 5)

        res = ds.inner_tile_size(shape, pos, size)
        np.testing.assert_array_equal([(3, 0), (0, 0)], res[-1])

    def test_get_padding_size_3(self):

        shape = c(7, 11, 7)
        pos = c(5, 5, 5)
        size = c(3, 5, 3)
        res = ds.inner_tile_size(shape, pos, size)
        np.testing.assert_array_equal([(0, 1), (0, 0), (0, 1)], res[-1])

    def test_get_padding_size_4(self):

        shape = c(7, 11, 7, 1)
        pos = c(5, 5, 5, 0)
        size = c(3, 5, 3, 1)
        res = ds.inner_tile_size(shape, pos, size)
        np.testing.assert_array_equal([(0, 1), (0, 0), (0, 1), (0, 0)],
                                      res[-1])

    def test_inner_tile_size_1(self):

        shape = c(7, 11)
        pos = c(-4, 8)
        size = c(16, 9)

        pos_val = c(0, 5)
        size_val = c(7, 6)
        pos_tile_val = c(0, 3)
        pos_out, size_out, pos_tile, padding = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile_val, pos_tile)
        np.testing.assert_array_equal(padding, [(4, 5), (0, 6)])

    def test_inner_tile_size_2(self):

        shape = c(7, 11)
        pos = c(-4, -1)
        size = c(5, 8)

        pos_val = c(0, 0)
        size_val = c(4, 7)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 0))
        np.testing.assert_array_equal(pd, [(4, 0), (1, 0)])

    def test_inner_tile_size_3(self):

        shape = c(7, 11)
        pos = c(-2, -3)
        size = c(3, 4)

        pos_val = c(0, 0)
        size_val = c(2, 3)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 0))
        np.testing.assert_array_equal(pd, [(2, 0), (3, 0)])

    def test_inner_tile_size_4(self):

        shape = c(7, 11)
        pos = c(-5, 2)
        size = c(6, 4)

        pos_val = c(0, 2)
        size_val = c(5, 4)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 0))
        np.testing.assert_array_equal(pd, [(5, 0), (0, 0)])

    def test_inner_tile_size_5(self):

        shape = c(7, 11)
        pos = c(2, -4)
        size = c(3, 8)

        pos_val = c(2, 0)
        size_val = c(3, 4)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 0))
        np.testing.assert_array_equal(pd, [(0, 0), (4, 0)])

    def test_inner_tile_size_6(self):

        shape = c(7, 11)
        pos = c(2, 9)
        size = c(3, 6)

        pos_val = c(2, 7)
        size_val = c(3, 4)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 2))
        np.testing.assert_array_equal(pd, [(0, 0), (0, 4)])

    def test_inner_tile_size_7(self):

        shape = c(7, 11)
        pos = c(2, 9)
        size = c(6, 5)

        pos_val = c(2, 8)
        size_val = c(5, 3)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 1))
        np.testing.assert_array_equal(pd, [(0, 1), (0, 3)])

    def test_inner_tile_size_8(self):

        shape = c(7, 11)
        pos = c(5, 9)
        size = c(5, 6)

        pos_val = c(4, 7)
        size_val = c(3, 4)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(tuple(pos_out), pos_val)
        np.testing.assert_array_equal(tuple(size_out), size_val)
        np.testing.assert_array_equal(tuple(pos_tile), (1, 2))
        np.testing.assert_array_equal(pd, [(0, 3), (0, 4)])

    def test_inner_tile_size_9(self):

        shape = c(7, 11)
        pos = c(-4, -1)
        size = c(6, 6)

        pos_val = c(0, 0)
        size_val = c(4, 5)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        np.testing.assert_array_equal(tuple(pos_out), pos_val)
        np.testing.assert_array_equal(tuple(size_out), size_val)
        np.testing.assert_array_equal(tuple(pos_tile), (0, 0))
        np.testing.assert_array_equal(pd, [(4, 0), (1, 0)])

    def test_inner_tile_size_10(self):

        shape = c(7, 11)
        pos = c(6, 7)
        size = c(4, 3)

        pos_val = c(4, 7)
        size_val = c(3, 3)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (2, 0))
        np.testing.assert_array_equal(pd, [(0, 3), (0, 0)])

    def test_inner_tile_size_11(self):

        shape = c(7, 11)
        pos = c(2, 2)
        size = c(3, 4)

        pos_val = c(2, 2)
        size_val = c(3, 4)

        pos_out, size_out, pos_tile, pd = \
            ds.inner_tile_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        np.testing.assert_array_equal(pos_out, pos_val)
        np.testing.assert_array_equal(size_out, size_val)
        np.testing.assert_array_equal(pos_tile, (0, 0))
        np.testing.assert_array_equal(pd, [(0, 0), (0, 0)])

    def test_is_padding(self):
        self.assertTrue(np.any([(2, 3), (0, 0)]))
        self.assertTrue(np.any([(2, 3), (20, 3)]))
        self.assertTrue(np.any([(0, 0), (3, 0)]))
        self.assertFalse(np.any([(0, 0), (0, 0)]))

    def test_get_weight_tile_for_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        img_nr = 2
        pos_czxy = (0, 0, 0)
        size_czxy = (1, 6, 4)
        label_value = 3
        mat = d._get_weights_tile(img_nr,
                                  pos_czxy, size_czxy, label_value)

        val = np.zeros(size_czxy)
        val[0, 0, 1] = 1
        val[0, 4, 1] = 1
        val[0, 5, 1] = 1

        np.testing.assert_array_equal(val, mat)

    def test_get_weight_tile_for_label_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        d.label_weights[3] = 1.2
        img_nr = 2
        pos_czxy = np.array((0, 0, 0))
        size_czxy = np.array((1, 6, 4))
        label_value = 3

        mat = d._get_weights_tile(img_nr,
                                  pos_czxy, size_czxy, label_value)

        val = np.zeros(size_czxy)
        print('valshae')
        print(val.shape)
        val[0, 0, 1] = 1.2
        val[0, 4, 1] = 1.2
        val[0, 5, 1] = 1.2
        pprint('mat')
        pprint(mat)
        pprint(val)
        self.assertTrue((val == mat).all())

    def test_load_label_counts_from_ilastik(self):
        img_path = os.path.join(base_path, '../test_data/ilastik')
        lbl_path = os.path.join(
            base_path, '../test_data/ilastik/ilastik-1.2.ilp')

        c = io_connector(img_path, lbl_path)
        d = Dataset(c)

        actual_counts = c.label_count_for_image(0)
        print(actual_counts)
        label_counts = d.load_label_counts()
        print(label_counts)

        assert_array_equal(label_counts[1], np.array([1]))
        assert_array_equal(label_counts[2], np.array([1]))
        assert_array_equal(label_counts[3], np.array([1]))

    def test_load_label_counts(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        t = d.load_label_counts()

        # labelcounts for each image for labelvalue 1
        expected_1 = np.array([4, 0, 0])
        # labelcounts for each image for labelvalue 2
        expected_2 = np.array([3, 0, 11])
        # labelcounts for each image for labelvalue 3
        expected_3 = np.array([3, 0, 3])

        assert_array_equal(expected_1, t[1])
        assert_array_equal(expected_2, t[2])
        assert_array_equal(expected_3, t[3])
        self.assertTrue(sorted(list(t.keys())), [1, 2, 3])

    def test_sync_label_counts(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        c1, c2 = c.split(1./3.)
        d1 = Dataset(c1)
        d2 = Dataset(c2)
        d1.sync_label_counts(d2)

        np.testing.assert_array_equal(set(d1.label_counts.keys()),
                                      set(d2.label_counts.keys()))

        assert_array_equal(d1.label_counts[1], np.array([4]))
        assert_array_equal(d1.label_counts[2], np.array([3]))
        assert_array_equal(d1.label_counts[3], np.array([3]))

        assert_array_equal(d2.label_counts[1], np.array([0, 0]))
        assert_array_equal(d2.label_counts[2], np.array([0, 11]))
        assert_array_equal(d2.label_counts[3], np.array([0, 3]))

        d1 = Dataset(c1)
        d2 = Dataset(c2)
        d2.sync_label_counts(d1)

        np.testing.assert_array_equal(set(d1.label_counts.keys()),
                                      set(d2.label_counts.keys()))

        assert_array_equal(d1.label_counts[1], np.array([4]))
        assert_array_equal(d1.label_counts[2], np.array([3]))
        assert_array_equal(d1.label_counts[3], np.array([3]))

        assert_array_equal(d2.label_counts[1], np.array([0, 0]))
        assert_array_equal(d2.label_counts[2], np.array([0, 11]))
        assert_array_equal(d2.label_counts[3], np.array([0, 3]))

    def test_set_label_weight(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        d.label_weights[3] = 0.7

        self.assertTrue(
            (d.label_weights[3] ==
             np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])).all())
        self.assertTrue((d.label_weights[1] == np.array([1, 1, 1, 1])).all())

    def test_equalize_label_weights(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        d.equalize_label_weights()

        print(d.label_weights)

        val = {1: 0.5121951219512196,
               2: 0.14634146341463417,
               3: 0.34146341463414637}

        np.testing.assert_array_equal(d.label_weights, val)

    @pytest.mark.skipif(sys.platform != 'linux', reason="Linux tests")
    def test__augment_tile(self):

        '''
        im =
        [[ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  3.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]]


         tile without augmentation
         tile =
          [[ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 1.  1.  1.  1.  3.  1.  1.  1.  1.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]]

        augmented tile afer 45 degree rotation
        tile_rot =
            [[ 1.  1.  0.  0.  0.  0.  0.  0.  2.]
             [ 0.  1.  1.  0.  0.  0.  0.  2.  2.]
             [ 0.  0.  1.  1.  0.  0.  2.  2.  0.]
             [ 0.  0.  0.  1.  1.  2.  2.  0.  0.]
             [ 0.  0.  0.  0.  3.  3.  0.  0.  0.]
             [ 0.  0.  0.  2.  2.  1.  1.  0.  0.]
             [ 0.  0.  2.  2.  0.  0.  1.  1.  0.]
             [ 0.  2.  2.  0.  0.  0.  0.  1.  1.]
             [ 2.  2.  0.  0.  0.  0.  0.  0.  1.]]

        '''

        def get_tile_func(pos=None, size=None, img=None):
            return img[tuple(get_tile_meshgrid(img.shape, pos, size))]

        val = np.array(
            [[[[1., 1., 0., 0., 0., 0., 0., 0., 2.],
               [0., 1., 1., 0., 0., 0., 0., 2., 2.],
               [0., 0., 1., 1., 0., 0., 2., 2., 0.],
               [0., 0., 0., 1., 1., 2., 2., 0., 0.],
               [0., 0., 0., 0., 3., 3., 0., 0., 0.],
               [0., 0., 0., 2., 2., 1., 1., 0., 0.],
               [0., 0., 2., 2., 0., 0., 1., 1., 0.],
               [0., 2., 2., 0., 0., 0., 0., 1., 1.],
               [2., 2., 0., 0., 0., 0., 0., 0., 1.]]]])

        im = np.zeros((1, 1, 15, 15))
        im[0, 0, 7, :] = 1
        im[0, 0, :, 7] = 2
        im[0, 0, 7, 7] = 3

        pos = np.array((0, 0, 3, 3))
        size = np.array((1, 1, 9, 9))
        tile = get_tile_func(pos=pos, size=size, img=im)

        tile_rot = ds._augment_tile(im.shape, pos, size,
                                    get_tile_func,
                                    augment_params={'rotation_angle': 45,
                                                    'shear_angle': 0},
                                    **{'img': im})

        print(im)
        print(tile)
        print(tile_rot)
        print(val)
        assert_array_equal(val, tile_rot)

    @pytest.mark.skipif(sys.platform != 'linux', reason="Linux tests")
    def test_augment_tile_2(self):

        '''
        im =
        [[ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  3.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]]

        '''

        def get_tile_func(pos=None, size=None, img=None):
            return img[tuple(get_tile_meshgrid(img.shape, pos, size))]

        val = np.array([[[[3.]]]])

        im = np.zeros((1, 1, 15, 15))
        im[0, 0, 7, :] = 1
        im[0, 0, :, 7] = 2
        im[0, 0, 7, 7] = 3

        pos = np.array((0, 0, 7, 7))
        size = np.array((1, 1, 1, 1))

        tile_rot = ds._augment_tile(im.shape, pos, size,
                                    get_tile_func,
                                    augment_params={'rotation_angle': 45,
                                                    'shear_angle': 0},
                                    **{'img': im})

        np.testing.assert_equal(val, tile_rot)

    def test_augment_tile_simple(self):

        '''
        im =
        [[ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  3.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]]

        '''

        def get_tile_func(pos=None, size=None, img=None):
            return img[tuple(get_tile_meshgrid(img.shape, pos, size))]

        val_ud = \
            np.array([[[[0., 0., 0., 0., 3., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 3., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 3., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 3., 0., 0., 0., 0.],
                        [2., 2., 2., 2., 5., 1., 1., 1., 1.],
                        [0., 0., 0., 0., 4., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 4., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 4., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 4., 0., 0., 0., 0.]]]])

        im = np.zeros((1, 1, 15, 15))
        im[0, 0, 7, :7] = 1
        im[0, 0, 7, 8:] = 2
        im[0, 0, :7, 7] = 3
        im[0, 0, 8:, 7] = 4
        im[0, 0, 7, 7] = 5

        pos = np.array((0, 0, 3, 3))
        size = np.array((1, 1, 9, 9))

        tile_ud = ds._augment_tile(
                    im.shape, pos, size,
                    get_tile_func,
                    augment_params={'flipud': True}, **{'img': im})
        assert_array_equal(tile_ud, val_ud)

    def test_multichannel_pixel_tile_1(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        img = 2
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 4, 3)
        channels = [1]

        val = np.array(
            [[[[102, 89, 82],
               [81, 37, 43],
               [87, 78, 68],
               [107, 153, 125]]]])

        tile = d.multichannel_pixel_tile(img, pos_zxy, size_zxy, channels)

        self.assertTrue((tile == val).all())

    def test_multichannel_pixel_tile_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        img = 2
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 4, 3)
        channels = [1]
        pd = (0, 2, 3)

        val = np.array(
            [[[[43, 37, 81, 81, 37, 43, 78, 78, 43],
              [82, 89, 102, 102, 89, 82, 87, 87, 82],
              [82, 89, 102, 102, 89, 82, 87, 87, 82],
              [43, 37, 81, 81, 37, 43, 78, 78, 43],
              [68, 78, 87, 87, 78, 68, 73, 73, 68],
              [125, 153, 107, 107, 153, 125, 82, 82, 125],
              [161, 180, 121, 121, 180, 161, 106, 106, 161],
              [147, 143, 111, 111, 143, 147, 123, 123, 147]]]])

        tile = d.multichannel_pixel_tile(img, pos_zxy, size_zxy, channels,
                                         pixel_padding=pd)

        self.assertTrue((tile == val).all())

    def test_multichannel_pixel_tile_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        img = 2
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 4, 3)
        channels = [1]
        pd = (0, 2, 3)

        val = np.array(
            [[[[73, 73, 43, 89, 89, 102, 81, 87, 78],
               [82, 68, 78, 37, 102, 102, 102, 37, 68],
               [161, 153, 78, 87, 81, 102, 89, 82, 78],
               [180, 180, 107, 87, 87, 37, 82, 87, 87],
               [143, 121, 121, 107, 78, 68, 78, 87, 87],
               [111, 111, 121, 180, 125, 73, 73, 78, 82],
               [121, 111, 143, 161, 106, 82, 73, 68, 43],
               [121, 180, 147, 123, 106, 106, 125, 68, 78]]]])

        tile = d.multichannel_pixel_tile(
            img, pos_zxy, size_zxy, channels,
            pixel_padding=pd,  augment_params={'rotation_angle': 45})

        self.assertTrue((tile == val).all())

    def test_channels_are_consistent(self):

        data_dir = os.path.join(base_path, '../test_data/cellvoyager')
        c = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        d = Dataset(c)
        is_consistent, channel_cnt = d.channels_are_consistent()
        assert is_consistent
        assert channel_cnt == [4]

        # check if assertion is raised for incomplete dataset
        data_dir_2 = os.path.join(base_path,
                                  '../test_data/cellvoyager_incomplete')
        c = CellvoyConnector(data_dir_2,
                             os.path.join(data_dir, 'labels_1.ilp'))
        self.assertRaises(AssertionError, Dataset, c)

    def test_multichannel_pixel_tile_cellvoy(self):

        data_dir = os.path.join(base_path, '../test_data/cellvoyager')
        c = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        d = Dataset(c)

        img = 0
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 1000, 992)
        size_zxy = (1, 500, 500)
        channels = [0, 1, 2, 3]
        pd = (0, 0, 0)

        tile = d.multichannel_pixel_tile(
            img, pos_zxy, size_zxy, channels,
            pixel_padding=pd,  augment_params={'rotation_angle': 45})

        assert not np.array_equal(tile[0, :, :, :], tile[3, :, :, :])
        assert not np.array_equal(tile[1, :, :, :], tile[3, :, :, :])
        assert not np.array_equal(tile[2, :, :, :], tile[3, :, :, :])

    def test_training_tile_1(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        img = 2
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 4, 3)
        channels = [1]
        labels = [2, 3]

        tr = d.training_tile(img, pos_zxy, size_zxy, channels, labels,
                             pixel_padding=(0, 1, 2))

        np.testing.assert_array_equal(tr.pixels.shape, (1, 1, 6, 7))
        np.testing.assert_array_equal(tr.weights.shape, (2, 1, 4, 3))

    def test_training_tile_cellvoy(self):

        data_dir = os.path.join(base_path, '../test_data/cellvoyager')
        c = CellvoyConnector(data_dir, os.path.join(data_dir, 'labels_1.ilp'))
        d = Dataset(c)

        img = 0
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 4, 3)
        channels = [0, 1, 2, 3]
        labels = [1, 2]

        d.training_tile(img, pos_zxy, size_zxy, channels, labels,
                        pixel_padding=(0, 1, 2))

    def test_random_training_tile(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)
        channels = [0, 1, 2]

        pixel_shape_val = (3, 3, 7, 8)
        weight_shape_val = (3, 1, 3, 4)

        # mapping [{91: 1, 109: 2, 150: 3}]
        labels_val = [1, 2, 3]
        tile = d.random_training_tile(
            size, channels, pixel_padding=pad,
            augment_params={'rotation_angle': 45})

        np.testing.assert_array_equal(tile.pixels.shape, pixel_shape_val)
        np.testing.assert_array_equal(tile.channels, channels)
        np.testing.assert_array_equal(tile.weights.shape, weight_shape_val)
        np.testing.assert_array_equal(tile.labels, labels_val)

    def test_get_label_probs(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        res = d._get_label_probs(label_value=None)
        assert_array_almost_equal(res, [0.416667, 0., 0.583333])

        res = d._get_label_probs(label_value=1)
        assert_array_almost_equal(res, [1., 0., 0.])

    def test_random_pos_izxy(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        np.random.seed(42)
        img_nr, pos_zxy = d._random_pos_izxy(label_value=1,
                                             tile_size_zxy=(1, 2, 2))

        assert_array_equal(img_nr, 0)
        assert_array_equal(pos_zxy, [2, 7, 20])

        # this tile has same size as image, pos must always be 0
        img_nr, pos_zxy = d._random_pos_izxy(label_value=1,
                                             tile_size_zxy=(3, 40, 26))
        assert_array_equal(img_nr, 0)
        assert_array_equal(pos_zxy, [0, 0, 0])

        # this tile is larger than the image
        with self.assertRaises(AssertionError):
            d._random_pos_izxy(label_value=1,
                               tile_size_zxy=(3, 40, 27))
        np.random.seed(None)

    def test_init_dataset_ilastik(self):
        p = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(p, 'images')
        label_path = os.path.join(p, 'x15_y10_z2_c4_classes2.ilp')

        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)
        self.assertEqual(d.n_images, 1)
        self.assertEqual(list(d.label_counts.keys()), [1, 2])  # label values

        assert_array_equal(d.label_counts[1], np.array([5]))
        assert_array_equal(d.label_counts[2], np.array([4]))

    def test_random_training_tile_by_polling_ilastik(self):

        p = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(p, 'images')
        label_path = os.path.join(p, 'x15_y10_z2_c4_classes2.ilp')

        size = (1, 1, 1)
        channels = [0, 1, 2, 3]
        labels = set([1, 2])
        ensure_labelvalue = 2

        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        np.random.seed(43)
        training_tile = d._random_training_tile_by_polling(
                                        size,
                                        channels,
                                        labels,
                                        ensure_labelvalue=ensure_labelvalue)
        print(training_tile)

        weights_val = np.array([[[[0.]]], [[[1.]]]])
        assert_array_equal(training_tile.weights, weights_val)

    def test_random_training_tile_by_polling(self):
        img_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/')

        size = (1, 4, 3)
        channels = [0, 1, 2]
        labels = set([1, 2, 3])
        ensure_labelvalue = 2

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        # weights_val = np.array(

        weights_val = np.array([[[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]],
                                [[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [1., 1., 1.],
                                  [1., 1., 1.]]],


                                [[[1., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]]])

        np.random.seed(43)
        training_tile = d._random_training_tile_by_polling(
                            size, channels, labels,
                            ensure_labelvalue=ensure_labelvalue)

        assert_array_equal(training_tile.weights, weights_val)

        weights_val = np.array([[[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.]]],
                                [[[0., 0., 0.],
                                  [0., 1., 1.],
                                  [0., 1., 1.],
                                  [0., 0., 0.]]],
                                [[[0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 0., 0.],
                                  [0., 1., 0.]]]])

        training_tile = d._random_training_tile_by_polling(
                            size, channels, labels,
                            ensure_labelvalue=None)

        assert_array_equal(training_tile.weights, weights_val)
        np.random.seed(None)
