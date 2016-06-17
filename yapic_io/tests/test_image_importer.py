from unittest import TestCase
import os

import numpy as np
import yapic.image_importers as ip
import logging
logger = logging.getLogger(os.path.basename(__file__))

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
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (1, 2, 6, 4))
        

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
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

