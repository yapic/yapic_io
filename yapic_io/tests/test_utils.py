from unittest import TestCase
import os
import numpy as np

from pprint import pprint

import yapic_io.utils as ut


class TestUtils(TestCase):

    def test_compute_pos(self):
        shape = (8, 5)
        size = (5, 2)

        res = ut.compute_pos(shape, size)
        val = [(0, 0), (0, 2), (0, 3), (3, 0), (3, 2), (3, 3)]
        self.assertEqual(res, val)



    def test_compute_pos_2(self):
        shape = (6, 4)
        size = (2, 2)

        res = ut.compute_pos(shape, size)
        print(res)
        val = [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)]
        self.assertEqual(val, res)


    def test_assign_slice_by_slice(self):
        vol = np.array([[[ 0., 0., 0.],
                         [ 0., 1., 2.]],
                        [[ 0., 0., 0.],
                         [ 0., 1., 2.]],
                        [[ 0., 0., 0.],
                         [ 0., 1., 2.]]])

        val = np.array([[[ 0., 0., 0.],
                         [ 0., 3., 4.]],
                        [[ 0., 0., 0.],
                         [ 0., 5., 6.]],
                        [[ 0., 0., 0.],
                         [ 0., 7., 8.]]])

        # vol = np.zeros((3, 2, 3))

        # vol[0, 1, 1] = 1
        # vol[0, 1, 2] = 2
        # vol[1, 1, 1] = 1
        # vol[1, 1, 2] = 2
        # vol[2, 1, 1] = 1
        # vol[2, 1, 2] = 2

        d = [{1:3, 2: 4}, {1:5, 2: 6}, {1:7, 2: 8}]

        out = ut.assign_slice_by_slice(d, vol)
        pprint(out)
        self.assertTrue(np.array_equal(out, val))


    def test_compute_str_dist_matrix(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']

        mat, a, b = ut.compute_str_dist_matrix(a, b)
        print(mat)
        print(a)
        print(b)


    def test_find_best_matching_pairs(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']
        val = [['hund', 'hund'], ['katze', 'kater'], ['maus', 'mauser']]
        pairs = ut.find_best_matching_pairs(a, b)
        print(pairs)
        self.assertEqual(pairs, val)

        #with Nones
        a = ['img_6width4height3slices_rgb.tif', 'img_40width26height3slices_rgb.tif', 'img_40width26height6slices_rgb.tif']
        b = ['lbl_6width4height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif']
        val = [['img_6width4height3slices_rgb.tif', 'lbl_6width4height3slices_rgb.tif']\
            , ['img_40width26height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif']\
            , ['img_40width26height6slices_rgb.tif', None]]
        pairs = ut.find_best_matching_pairs(a, b)
        print(pairs)
        self.assertEqual(pairs, val)















