from unittest import TestCase
import os

import numpy as np
from yapic_io.tiffconnector import Tiffconnector
import logging
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestTiffconnector(TestCase):
    def test_load_filenames(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'pypath')

        img_filenames = [\
                ('6width4height3slices_rgb.tif',)\
                , ('6width4height3slices_rgb.tif',)\
                , ('40width26height3slices_rgb.tif',)\
                , ('40width26height6slices_rgb.tif',)\
                ]

        c.load_img_filenames()
        print('c filenames:')
        print(c.filenames)

        self.assertEqual(set(img_filenames), set(c.filenames))




        # path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        # dims = ip.get_tiff_image_dimensions(path)
        # self.assertEqual(dims, (3, 3, 6, 4))

        # path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')
        # dims = ip.get_tiff_image_dimensions(path)
        # self.assertEqual(dims, (3, 2, 6, 4))

        # path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        # dims = ip.get_tiff_image_dimensions(path)
        # self.assertEqual(dims, (3, 1, 6, 4))

        # path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        # dims = ip.get_tiff_image_dimensions(path)
        # self.assertEqual(dims, (1, 2, 6, 4))
        # 