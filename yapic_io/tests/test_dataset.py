from unittest import TestCase
import os

import numpy as np
from yapic_io.tiffconnector import Tiffconnector
from yapic_io.dataset import Dataset

import yapic_io.dataset as ds

import logging
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestDataset(TestCase):
    def test_n_images(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        d = Dataset(c)

        self.assertEqual(d.n_images, 3)
        # img_filenames = [\
        #         ('6width4height3slices_rgb.tif',)\
        #         , ('40width26height3slices_rgb.tif',)\
        #         , ('40width26height6slices_rgb.tif',)\
        #         ]

        
        

        


    def test_get_padding_size_1(self):

        shape = (7, 11)
        pos = (5, 5)
        size = (3, 5)
        res = ds.get_padding_size(shape, pos, size)
        self.assertEqual([(0, 1), (0, 0)], res)


    def test_get_padding_size_2(self):

        shape = (7, 11)
        pos = (-3, 0)
        size = (3, 5)

        res = ds.get_padding_size(shape, pos, size)
        self.assertEqual([(3,0),(0,0)], res) 

    def test_get_dist_to_upper_img_edge(self):


        pos = (3, 1)
        shape = (5, 4)
        val = np.array([1, 2])
        res = ds.get_dist_to_upper_img_edge(shape, pos)
        self.assertTrue((val == res).all())


        pos = (4, 3)
        shape = (5, 4)
        val = np.array([0, 0])
        res = ds.get_dist_to_upper_img_edge(shape, pos)
        self.assertTrue((val == res).all())

    # def test_pos_shift_for_padding(self):

    #     pos = (4, 3)
    #     shape = (5, 4)
    #     size=(3, 3)
    #     res = ds.pos_shift_for_padding(shape, pos, size)
    #     dist = ds.get_dist_to_upper_img_edge(shape, pos)
    #     print(dist)
    #     print(res)
    #     self.assertTrue(False)



        
