from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset

from yapic_io.training_batch import TrainingBatch
from pprint import pprint
import numpy as np
import tempfile
from numpy.testing import assert_array_equal, assert_array_almost_equal

base_path = os.path.dirname(__file__)

class TestTrainingBatch(TestCase):

    def test_get_ilastik_weights(self):
        img_path = os.path.join(base_path,
                                '../test_data/ilastik')
        lbl_path = os.path.join(base_path,
                                '../test_data/ilastik/ilastik-1.2.ilp')

        c = IlastikConnector(img_path, lbl_path)

        d = Dataset(c)

        size = (8, 2, 4)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)
        m.augment_by_flipping(False)
        mini = next(m)
        weights = mini.weights()

        # label 1 at position (4,1,3)
        self.assertTrue(weights[0, 0, 4, 1, 3] == 1)

        # label 2 at position (1,1,1)
        self.assertTrue(weights[0, 1, 1, 1, 1] == 1)

        # label 3 at position (1,1,1)
        self.assertTrue(weights[0, 2, 7, 1, 1] == 1)

    def test_get_ilastik_weights2(self):

        pth = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(pth, 'images')
        lbl_path = os.path.join(pth, 'x15_y10_z2_c4_classes2.ilp')
        c = IlastikConnector(img_path, lbl_path)

        d = Dataset(c)

        size = (2, 15, 10)  # this is the size of the whole image
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)
        m.augment_by_flipping(False)
        mini = next(m)
        weights = mini.weights()

        print(weights[0, :, :, :, :])

        # all samples of the batch should be identical,
        # because tilesize=imsize
        assert_array_equal(weights[0, :, :, :, :], weights[1, :, :, :, :])

        # weight positions from labelvalue 1 and 2
        pos = [[0, 0, 2, 1], [0, 0, 2, 1], [0, 0, 8, 6], [0, 0, 9, 6],
               [0, 1, 4, 3],
               [1, 0, 2, 2], [1, 0, 3, 2], [1, 0, 8, 7], [1, 0, 9, 7]]

        for p in pos:
            self.assertEqual(weights[0, p[0], p[1], p[2], p[3]], 1)
            self.assertEqual(weights[1, p[0], p[1], p[2], p[3]], 1)

    def test_pixel_format_is_float(self):
        p = os.path.join(base_path, '../test_data/ilastik/dimensionstest')
        img_path = os.path.join(p, 'images')
        label_path = os.path.join(p, 'x15_y10_z2_c4_classes2.ilp')

        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        size = (2, 4, 3)
        pad = (0, 0, 0)
        m = TrainingBatch(d, size, padding_zxy=pad)
        mini = next(m)
        p = mini._pixels
        self.assertTrue(np.issubdtype(p.dtype, np.float))

    def test_random_tile(self):

        img_path = os.path.join(base_path,
                                '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)

        m = TrainingBatch(d, size, padding_zxy=pad)

        m._random_tile(for_label=1)

    def test_getitem(self):

        # define data locations
        pixel_image_dir = os.path.join(base_path,
                                       '../test_data/tiffconnector_1/im/')
        label_image_dir = os.path.join(base_path,
                                       '../test_data/tiffconnector_1/labels/')
        savepath = tempfile.TemporaryDirectory()

        tile_size = (1, 5, 4)  # size of network output layer in zxy
        # padding of network input layer in zxy, in respect to output layer
        padding = (0, 2, 2)

        c = TiffConnector(pixel_image_dir,
                          label_image_dir,
                          savepath=savepath.name)
        m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)

        for counter, mini in enumerate(m):
            # shape is (6, 3, 1, 5, 4):
            # batchsize 6, 3 label-classes, 1 z, 5 x, 4 y
            weights = mini.weights()

            # shape is (6, 3, 1, 9, 8):
            # batchsize 6, 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
            pixels = mini.pixels()
            self.assertEqual(weights.shape, (3, 3, 1, 5, 4))
            self.assertEqual(pixels.shape, (3, 3, 1, 9, 8))

            # apply training on mini.pixels and mini.weights goes here

            if counter > 10:  # m is infinite
                break

    def test_getitem_multichannel_labels(self):

        # define data loacations
        pixel_image_dir = os.path.join(base_path,
                                       '../test_data/tiffconnector_1/im/')
        label_image_dir = os.path.join(
                        base_path,
                        '../test_data/tiffconnector_1/labels_multichannel/')
        savepath = tempfile.TemporaryDirectory()

        tile_size = (1, 5, 4)  # size of network output layer in zxy
        # padding of network input layer in zxy, in respect to output layer
        padding = (0, 2, 2)

        # make training_batch mb and prediction interface p with
        # TiffConnector binding
        c = TiffConnector(pixel_image_dir, label_image_dir,
                          savepath=savepath.name)
        m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)

        for counter, mini in enumerate(m):
            # shape is (6, 6, 1, 5, 4):
            # batchsize 6 , 6 label-classes, 1 z, 5 x, 4 y
            weights = mini.weights()

            # shape is (6, 3, 1, 9, 8):
            # batchsize 6, 6 channels, 1 z, 9 x, 4 y (more xy due to padding)
            pixels = mini.pixels()
            self.assertEqual(weights.shape, (6, 6, 1, 5, 4))
            self.assertEqual(pixels.shape, (6, 3, 1, 9, 8))

            # apply training on mini.pixels and mini.weights goes here

            if counter > 10:  # m is infinite
                break

    def test_normalize_zscore(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 2, 2)
        pad = (0, 0, 0)

        batchsize = 2
        nr_channels = 3
        nz = 1
        nx = 4
        ny = 5

        m = TrainingBatch(d, size, padding_zxy=pad)

        m.set_normalize_mode('local_z_score')

        pixels = np.zeros((batchsize, nr_channels, nz, nx, ny))
        pixels[:, 0, :, :, :] = 1
        pixels[:, 1, :, :, :] = 2
        pixels[:, 2, :, :, :] = 3

        p_norm = m._normalize(pixels)
        self.assertTrue((p_norm == 0).all())

        # add variation
        pixels[:, 0, 0, 0, 0] = 2
        pixels[:, 0, 0, 0, 1] = 0

        pixels[:, 1, 0, 0, 0] = 3
        pixels[:, 1, 0, 0, 1] = 1

        pixels[:, 2, 0, 0, 0] = 4
        pixels[:, 2, 0, 0, 1] = 2

        p_norm = m._normalize(pixels)

        assert_array_equal(p_norm[:, 0, :, :, :], p_norm[:, 1, :, :, :])
        assert_array_equal(p_norm[:, 0, :, :, :], p_norm[:, 2, :, :, :])

        val = np.array([[[3.16227766, -3.16227766, 0., 0., 0.],
                         [0.,         0.,          0., 0., 0.],
                         [0.,         0.,          0., 0., 0.],
                         [0.,         0.,          0., 0., 0.]]])

        assert_array_almost_equal(val, p_norm[0, 0, :, :, :])

    def test_normalize_global(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 2, 2)
        pad = (0, 0, 0)

        batchsize = 2
        nr_channels = 3
        nz = 1
        nx = 4
        ny = 5

        val = np.array(
         [[[0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333],
           [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333],
           [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333],
           [0.33333333, 0.33333333, 0.33333333, 0.33333333, 0.33333333]],
          [[0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
           [0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
           [0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667],
           [0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667]],
          [[1.,         1.,         1.,         1.,         1.],
           [1.,         1.,         1.,         1.,         1.],
           [1.,         1.,         1.,         1.,         1.],
           [1.,         1.,         1.,         1.,         1.]]])

        m = TrainingBatch(d, size, padding_zxy=pad)
        m.set_normalize_mode('global', minmax=[0, 3])

        pixels = np.zeros((batchsize, nr_channels, nz, nx, ny))
        pixels[:, 0, :, :, :] = 1
        pixels[:, 1, :, :, :] = 2
        pixels[:, 2, :, :, :] = 3

        p_norm = m._normalize(pixels)

        pprint(p_norm)
        print(pixels.shape)
        print(p_norm.shape)

        assert_array_almost_equal(val, p_norm[0, :, 0, :, :])

    def test_set_augmentation(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)

        m = TrainingBatch(d, size, padding_zxy=pad)

        self.assertEqual(m.augmentation, {'flip'})

        m.augment_by_rotation(True)
        self.assertEqual(m.augmentation, {'flip', 'rotate'})
        self.assertEqual(m.rotation_range, (-45, 45))

        m.augment_by_shear(True)
        self.assertEqual(m.augmentation, {'flip', 'rotate', 'shear'})
        self.assertEqual(m.shear_range, (-5, 5))

        m.augment_by_flipping(False)
        self.assertEqual(m.augmentation, {'rotate', 'shear'})


    def test_set_pixel_dimension_order(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)

        m = TrainingBatch(d, size, padding_zxy=pad)

        m.set_pixel_dimension_order('bczxy')
        self.assertEqual([0, 1, 2, 3, 4], m.pixel_dimension_order)

        m.set_pixel_dimension_order('bzxyc')
        self.assertEqual([0, 4, 1, 2, 3], m.pixel_dimension_order)


    def test_get_pixels_dimension_order(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path,
                                  '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (2, 5, 4)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)

        b = next(m)

        p = m.pixels()
        w = m.weights()
        self.assertEqual(p.shape, (3, 3, 2, 5, 4))
        self.assertEqual(w.shape, (3, 3, 2, 5, 4))

        m.set_pixel_dimension_order('bczxy')
        p = m.pixels()
        w = m.weights()
        self.assertEqual(p.shape, (3, 3, 2, 5, 4))
        self.assertEqual(w.shape, (3, 3, 2, 5, 4))

        m.set_pixel_dimension_order('bzxyc')
        p = m.pixels()
        w = m.weights()
        self.assertEqual(p.shape, (3, 2, 5, 4, 3))
        self.assertEqual(w.shape, (3, 2, 5, 4, 3))

    def test_tile_positions_decay(self):

        img_path = os.path.join(
            base_path,
            '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        label_path = os.path.join(
            base_path,
            '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        size = (2, 6, 4)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)

        n_pos_lbl_1 = len(m.tile_pos_for_label[1])
        n_pos_lbl_2 = len(m.tile_pos_for_label[2])

        for _ in range(800):
            next(m)

        n_pos_lbl_1_after = len(m.tile_pos_for_label[1])
        n_pos_lbl_2_after = len(m.tile_pos_for_label[2])

        self.assertTrue(n_pos_lbl_1 > n_pos_lbl_1_after)
        self.assertTrue(n_pos_lbl_2 > n_pos_lbl_2_after)

    def test_remove_unlabeled_tiles(self):

        img_path = os.path.join(
            base_path,
            '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        label_path = os.path.join(
            base_path,
            '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        size = (2, 6, 4)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)

        n_pos_lbl_1 = len(m.tile_pos_for_label[1])
        n_pos_lbl_2 = len(m.tile_pos_for_label[2])

        m.remove_unlabeled_tiles()

        n_pos_lbl_1_after = len(m.tile_pos_for_label[1])
        n_pos_lbl_2_after = len(m.tile_pos_for_label[2])

        self.assertTrue(n_pos_lbl_1 > n_pos_lbl_1_after)
        self.assertTrue(n_pos_lbl_2 > n_pos_lbl_2_after)

    def test_split(self):

        img_path = os.path.join(
            base_path,
            '../test_data/ilastik/pixels_ilastik-multiim-1.2')
        label_path = os.path.join(
            base_path,
            '../test_data/ilastik/ilastik-multiim-1.2.ilp')
        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 1, 1)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)
        m2 = m.split(0.3)

        self.assertTrue(
            len(m.tile_pos_for_label[1]) > len(m2.tile_pos_for_label[1]))

        self.assertTrue(
            len(m.tile_pos_for_label[2]) > len(m2.tile_pos_for_label[2]))

        # check for no overlap
        self.assertEqual(
            0,
            len(set(m.tile_pos_for_label[1]) & set(m2.tile_pos_for_label[1])))

        self.assertEqual(
            0,
            len(set(m2.tile_pos_for_label[1]) & set(m.tile_pos_for_label[1])))

        assert False

    def test_shape_data(self):

        import logging
        logging.basicConfig(level=logging.INFO)

        img_path = os.path.join(
            base_path,
            '../test_data/shapes/pixels/*')
        label_path = os.path.join(
            base_path,
            '../test_data/shapes/labels.ilp')
        c = IlastikConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 50, 50)
        pad = (0, 0, 0)

        m = TrainingBatch(d, size, padding_zxy=pad)

        m2 = m.split(0.001)

        assert False
