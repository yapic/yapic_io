import tempfile
from unittest import TestCase
import os
from yapic_io.connector import io_connector
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from yapic_io import TiffConnector, Dataset, PredictionBatch
from bigtiff import Tiff
from yapic_io.ilastik_connector import IlastikConnector


base_path = os.path.dirname(__file__)


class TestPredictionBatch(TestCase):
    def test_computepos_1(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 1, 1)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        self.assertEqual(len(p._all_tile_positions), 6 * 4 * 3)
        tilepos = [(p[0], tuple(p[1])) for p in p._all_tile_positions]

        self.assertEqual(len(tilepos),
                         len(set(tilepos)))

    def test_computepos_2(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (3, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        val = [(0, (0, 0, 0))]
        for pos, valpos in zip(p._all_tile_positions, val):
            assert_array_equal(pos[1], np.array(valpos[1]))
            self.assertEqual(pos[0], valpos[0])

    def test_computepos_3(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (2, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        val = [(0, (0, 0, 0)), (0, (1, 0, 0))]
        for pos, valpos in zip(p._all_tile_positions, val):
            assert_array_equal(pos[1], np.array(valpos[1]))
            self.assertEqual(pos[0], valpos[0])

    def test_getitem_1(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        # batch size is 2, so the first 2 tiles go with the first batch
        # (size two), the third tile in in the second batch. the second
        # batch has only size 1 (is smaller than the specified batch size),
        # because it contains the rest.

        self.assertEqual(len(p), 2)
        self.assertEqual(p[0].pixels().shape, (2, 3, 1, 6, 4))
        self.assertEqual(p[1].pixels().shape, (1, 3, 1, 6, 4))

    def test_getitem_2(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 3

        p = PredictionBatch(d, batch_size, size)
        # batch size is 3, this means all3 tempplates fit in one batch

        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].pixels().shape, (3, 3, 1, 6, 4))

    def test_current_tile_positions(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')

        c = TiffConnector(img_path, label_path)
        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)

        val = [(0, (0, 0, 0)), (0, (1, 0, 0))]
        for pos, valpos in zip(p[0].current_tile_positions, val):
            assert_array_equal(pos[1], np.array(valpos[1]))
            self.assertEqual(pos[0], valpos[0])

        val = [(0, (2, 0, 0))]
        for pos, valpos in zip(p[1].current_tile_positions, val):
            assert_array_equal(pos[1], np.array(valpos[1]))
            self.assertEqual(pos[0], valpos[0])



    def test_put_probmap_data(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = tempfile.TemporaryDirectory()

        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((1, 2, 1, 6, 4))
        p[0].put_probmap_data(data)
        p[1].put_probmap_data(data)
        p[2].put_probmap_data(data)

    def test_put_probmap_data_2(self):
        img_path = os.path.join(
            base_path,
            '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = tempfile.TemporaryDirectory()

        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        size = (1, 2, 2)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        pixel_val = 0
        for mb in p:
            pixel_val += 10
            data = np.ones((1, 2, 1, 2, 2)) * pixel_val
            mb.put_probmap_data(data)

        pixelmap = Tiff.memmap_tcz(os.path.join(savepath.name,
                                   '6width4height3slices_rgb_class_1.tif'))
        # zslice 0
        val_0 = np.array([[10., 10., 30., 30., 50., 50.],
                          [10., 10., 30., 30., 50., 50.],
                          [20., 20., 40., 40., 60., 60.],
                          [20., 20., 40., 40., 60., 60.]])
        assert_array_almost_equal(pixelmap[0][0][0], val_0)

        # zslice 1
        val_1 = np.array([[70.,  70.,  90.,  90., 110., 110.],
                          [70.,  70.,  90.,  90., 110., 110.],
                          [80.,  80., 100., 100., 120., 120.],
                          [80.,  80., 100., 100., 120., 120.]])
        assert_array_almost_equal(pixelmap[0][0][1], val_1)

        # zslice 2
        val_2 = np.array([[130., 130., 150., 150., 170., 170.],
                          [130., 130., 150., 150., 170., 170.],
                          [140., 140., 160., 160., 180., 180.],
                          [140., 140., 160., 160., 180., 180.]])
        assert_array_almost_equal(pixelmap[0][0][2], val_2)

    def test_put_probmap_data_3(self):
        img_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels/*')
        savepath = tempfile.TemporaryDirectory()

        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        size = (1, 3, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2, 3, 1, 3, 4))
        p[0].put_probmap_data(data)

        data = np.ones((2, 3, 1, 3, 4))
        p[1].put_probmap_data(data)

        data = np.ones((2, 3, 1, 3, 4))
        p[2].put_probmap_data(data)

    def test_put_probmap_data_when_no_labels_available(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        savepath = tempfile.TemporaryDirectory()
        c = io_connector(img_path, '', savepath=savepath.name)
        d = Dataset(c)

        size = (1, 3, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2, 2, 1, 3, 4))
        p[0].put_probmap_data(data)

        data = np.ones((2, 2, 1, 3, 4))
        p[1].put_probmap_data(data)

        data = np.ones((2, 2, 1, 3, 4))
        p[2].put_probmap_data(data)

        val = ['40width26height3slices_rgb_class_1.tif',
               '40width26height3slices_rgb_class_2.tif']
        self.assertEqual(sorted(os.listdir(savepath.name)), val)

    def test_put_probmap_data_multichannel_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(
            base_path, '../test_data/tiffconnector_1/labels_multichannel/*')
        savepath = tempfile.TemporaryDirectory()

        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        original_labels = c.original_label_values_for_all_images()
        res = c.calc_label_values_mapping(original_labels)

        d = Dataset(c)

        size = (1, 3, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((1, 6, 1, 3, 4))
        p[0].put_probmap_data(data)


    def test_put_probmap_data_dimorder_zxyc(self):

        img_path = os.path.join(
            base_path,
            '../test_data/ilastik/dimensionstest/images/*')
        label_path = os.path.join(
            base_path,
            '../test_data/ilastik/dimensionstest/x15_y10_z2_c4_classes2.ilp')

        with tempfile.TemporaryDirectory() as tmpdirname:
            c = TiffConnector(img_path, 'some/path', savepath=tmpdirname)
            d = Dataset(c)

            size = (2, 3, 5) # zxy
            batch_size = 1

            # shape: 1 batch, 4 channels, 2 z, 3 x, 5 y

            p = PredictionBatch(d, batch_size, size)
            self.assertEqual((1, 4, 2, 3, 5), p[0].pixels().shape)
            self.assertEqual((1, 4, 2, 3, 5), p[1].pixels().shape)

            p.set_pixel_dimension_order('bzxyc')

            self.assertEqual((1, 2, 3, 5, 4), p[0].pixels().shape)
            self.assertEqual((1, 2, 3, 5, 4), p[1].pixels().shape)

            pixels = p[0].pixels()

            for tile in p:
                tile.put_probmap_data(pixels)


            # assert False








    def test_prediction_loop(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # mock classification function
            def classify(pixels, value):
                return np.ones(pixels.shape) * value

            # define data locations
            pixel_image_dir = os.path.join(
                base_path, '../test_data/tiffconnector_1/im/*.tif')
            label_image_dir = os.path.join(
                base_path, '../test_data/tiffconnector_1/labels/*.tif')
            savepath = tmpdirname

            tile_size = (1, 5, 4)  # size of network output layer in zxy
            padding = (0, 0, 0)  # padding of network input layer in zxy,
            # in respect to output layer

            # Make training_batch mb and prediction interface p with
            # TiffConnector binding.
            c = TiffConnector(pixel_image_dir,
                              label_image_dir, savepath=savepath)
            p = PredictionBatch(Dataset(c), 2, tile_size, padding_zxy=padding)

            self.assertEqual(len(p), 255)
            self.assertEqual(p.labels, {1, 2, 3})

            # classify the whole bound dataset
            for counter, item in enumerate(p):
                pixels = item.pixels()  # input for classifier
                mock_classifier_result = classify(pixels, counter)
                # pass classifier results for each class to data source
                item.put_probmap_data(mock_classifier_result)


    def test_pixel_dimensions(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        savepath = tempfile.TemporaryDirectory()
        c = io_connector(img_path, '', savepath=savepath.name)
        d = Dataset(c)

        size = (1, 5, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)[0]

        print(p.pixels().shape)
        self.assertEqual((2, 3, 1, 5, 4), p.pixels().shape)

        p.set_pixel_dimension_order('bzxyc')
        self.assertEqual((2, 1, 5, 4, 3), p.pixels().shape)
