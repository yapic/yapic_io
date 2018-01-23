from unittest import TestCase
import os
import tempfile

import numpy as np
import yapic_io.image_importers as ip
import logging
logger = logging.getLogger(os.path.basename(__file__))
from tifffile import imsave, imread
base_path = os.path.dirname(__file__)

class TestImageImports(TestCase):
    def test_get_tiff_image_dimensions(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 3, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 2, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 1, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path, zstack=True)
        self.assertEqual(dims, (1, 2, 6, 4))
        
    def test_imread_1(self):
        path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/40width26height3slices_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('2channel mask, three slices')
        print(im.shape)
        self.assertEqual(dims, (2, 3, 40, 26))  

    def test_imread_2(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('rgb, one z slice')
        print(im.shape)
        self.assertEqual(dims, (3, 1, 6, 4))

    def test_imread_3(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        im = ip.import_tiff_image(path, zstack=True)
        dims = im.shape
        print('grayscale, three z slices')
        print(im.shape)
        self.assertEqual(dims, (1, 3, 6, 4))

    def test_imread_3(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_1slice_2channels.tif')
        im = ip.import_tiff_image(path, multichannel=True)
        dims = im.shape
        print('grayscale, 1 slice, 2 channels')
        print(im.shape)
        self.assertEqual(dims, (2, 1, 6, 4))

    def test_imread_4(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_1slice_1channel.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('grayscale, 1 slice, 1 channel')
        print(im.shape)
        self.assertEqual(dims, (1, 1, 6, 4))

    def test_imread_5(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('rgb, 3 slice, 3 channel')
        print(im.shape)
        self.assertEqual(dims, (3, 3, 6, 4))                            

    def test_get_tiff_image_dimensions_4(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape


        
        self.assertEqual(dims, (3, 1, 6, 4))               



    def test_import_tiff_image(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        
        im = ip.import_tiff_image(path, zstack=True)
        self.assertEqual((1, 2, 6, 4), im.shape)
        
        



    def test_import_tiff_image_2(self):
        path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/6width4height3slices_rgb.tif')
        im = ip.import_tiff_image(path)
        print(im.shape)
        print(np.unique(im.flatten()))
        self.assertEqual((2, 3, 6, 4), im.shape)
