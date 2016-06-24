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
        c = Tiffconnector(img_path,'path/to/nowhere/')

        img_filenames = [\
                ('6width4height3slices_rgb.tif',)\
                , ('40width26height3slices_rgb.tif',)\
                , ('40width26height6slices_rgb.tif',)\
                ]

        c.load_img_filenames()
        print('c filenames:')
        print(c.filenames)

        self.assertEqual(set(img_filenames), set(c.filenames))


    def test_load_filenames_emptyfolder(self):
        img_path = os.path.join(base_path, '../test_data/empty_folder/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        #c.load_img_filenames()
        self.assertIsNone(c.filenames)



    

    def test_load_img_dimensions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        
        
        self.assertFalse(c.load_img_dimensions(-1))
        self.assertFalse(c.load_img_dimensions(4))   
        
        self.assertEqual(c.load_img_dimensions(0), (3, 3, 6, 4))   
        self.assertEqual(c.load_img_dimensions(1), (3, 6, 40, 26))
        self.assertEqual(c.load_img_dimensions(2), (3, 3, 40, 26))

    def test_load_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        
        im = c.load_image(0)
        print(im)
        
        self.assertEqual(im.shape, (3, 3, 6, 4))   

    def test_get_template(self):
        
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        

        image_nr = 0
        pos = (0, 0, 0, 0)
        size = (1, 1, 1, 2)
        im = c.load_image(0)
        tpl = c.get_template(image_nr, pos, size)
        val = np.empty(shape=size)
        val[0,0,0,0] = 151
        val[0,0,0,1] = 132
        val = val.astype(int)
        print(val)
        print(tpl)
        self.assertTrue((tpl == val).all())




