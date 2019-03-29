from unittest import TestCase
import yapic_io.utils as ut
import numpy as np

class TestUtils(TestCase):

    def test_compute_pos(self):
        shape = (8, 5)
        size = (5, 2)

        res = ut.compute_pos(shape, size)
        val = [(0, 0), (0, 2), (0, 3), (3, 0), (3, 2), (3, 3)]

        self.assertEqual(res, val)

        res = ut.compute_pos(shape, size, sliding_window=True)
        val = [(0, 0), (1, 0), (2, 0), (3, 0),
               (0, 1), (1, 1), (2, 1), (3, 1),
               (0, 2), (1, 2), (2, 2), (3, 2),
               (0, 3), (1, 3), (2, 3), (3, 3)]

        self.assertEqual(res, val)

    def test_compute_pos_2(self):
        shape = (6, 4)
        size = (2, 2)

        res = ut.compute_pos(shape, size)
        val = [(0, 0), (0, 2), (2, 0), (2, 2), (4, 0), (4, 2)]
        self.assertEqual(val, res)

        res = ut.compute_pos(shape, size, sliding_window=True)
        val = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
               (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
               (0, 2), (1, 2), (2, 2), (3, 2), (4, 2)]
        self.assertEqual(val, res)

    def test_compute_pos_3d(self):

        shape = (2, 4, 3)
        size = (2, 4, 3)

        val = [(0, 0, 0)]

        res = ut.compute_pos(shape, size)
        self.assertEqual(val, res)

        res = ut.compute_pos(shape, size, sliding_window=True)
        self.assertEqual(val, res)

    def test_compute_pos_3d_2(self):

        shape = (2, 4, 3)
        size = (1, 4, 3)

        val = [(0, 0, 0), (1, 0, 0)]

        res = ut.compute_pos(shape, size)
        self.assertEqual(val, res)

        res = ut.compute_pos(shape, size, sliding_window=True)
        self.assertEqual(val, res)


    def test_compute_pos_3d_3(self):

        shape = (6, 4, 3)
        size = (2, 4, 3)

        val = [(0, 0, 0), (2, 0, 0), (4, 0, 0)]

        res = ut.compute_pos(shape, size)
        self.assertEqual(val, res)

        val = [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0)]

        res = ut.compute_pos(shape, size, sliding_window=True)
        self.assertEqual(val, res)

    def test_compute_pos_large_tile(self):

        shape = (500, 500)
        size = (50,  50)

        res = ut.compute_pos(shape, size)

        print(res)

        res = ut.compute_pos(shape, size, sliding_window=True)

    def test_remove_overlapping_pos(self):

        pos = [(1, 1), (1, 3), (5, 4), (6, 2)]
        a = (3, 3)
        shape = (3, 2)

        is_overlap = ut.find_overlapping_tiles(a, pos, shape)

        np.testing.assert_array_equal(np.array([False, True, True, False]),
                                      is_overlap)

        pos = [(0, 1, 1), (0, 1, 3), (0, 5, 4), (0, 6, 2)]
        a = (0, 3, 3)
        shape = (1, 3, 2)

        is_overlap = ut.find_overlapping_tiles(a, pos, shape)

        np.testing.assert_array_equal(np.array([False, True, True, False]),
                                      is_overlap)

    def test_find_overlapping_tiles_4d(self):

        pos = [(0, 0, 0, 0),
               (0, 0, 1, 0),
               (0, 0, 2, 0),
               (0, 0, 3, 0)]

        shape = (1, 1, 50, 50)

        a = (1, 0, 374, 165)
        is_overlap = ut.find_overlapping_tiles(a, pos, shape)
        np.testing.assert_array_equal(np.array([False, False, False, False]),
                                      is_overlap)

        a = (0, 0, 4, 0)
        is_overlap = ut.find_overlapping_tiles(a, pos, shape)
        np.testing.assert_array_equal(np.array([True, True, True, True]),
                                      is_overlap)

        a = (1, 0, 4, 0)
        is_overlap = ut.find_overlapping_tiles(a, pos, shape)
        np.testing.assert_array_equal(np.array([False, False, False, False]),
                                      is_overlap)

    def test_segregate_pos(self):

        pos = [(1, 1), (1, 3), (3, 3), (5, 4), (6, 2), (10, 10), (20, 20)]
        shape = (3, 2)
        indices = [2, 5, 6]

        p1, p2 = ut.segregate_tile_pos(pos, shape, indices)

        v1 = [(1, 1), (6, 2)]
        v2 = [(3, 3), (10, 10), (20, 20)]

        self.assertEqual(p1, v1)
        self.assertEqual(p2, v2)

    def test_compute_str_dist_matrix(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']

        mat, a, b = ut._compute_str_dist_matrix(a, b)

    def test_find_best_matching_pairs(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']
        val = [['hund', 'hund'], ['katze', 'kater'], ['maus', 'mauser']]
        pairs = ut.find_best_matching_pairs(a, b)
        print(pairs)
        self.assertEqual(pairs, val)

        # with Nones
        a = ['img_6width4height3slices_rgb.tif',
             'img_40width26height3slices_rgb.tif',
             'img_40width26height6slices_rgb.tif']

        b = ['lbl_6width4height3slices_rgb.tif',
             'lbl_40width26height3slices_rgb.tif']

        val = [['img_6width4height3slices_rgb.tif',
                'lbl_6width4height3slices_rgb.tif'],
               ['img_40width26height3slices_rgb.tif',
                'lbl_40width26height3slices_rgb.tif'],
               ['img_40width26height6slices_rgb.tif',
                None]]

        pairs = ut.find_best_matching_pairs(a, b)
        self.assertEqual(pairs, val)
