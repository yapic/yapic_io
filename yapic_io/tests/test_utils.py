from unittest import TestCase
import os
import numpy as np 

from pprint import pprint

import yapic_io.utils as ut


class TestUtils(TestCase):
    
    
    def test_get_tile_meshgrid(self):

        shape = (10000, 30000)
        pos = (3, 3)
        size = (7000, 9000)

        res = ut.get_tile_meshgrid(shape, pos, size)

        #pprint(res)

        #self.assertTrue(False)

    
    


    def test_nest_list(self):
        t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        val = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        res = ut.nest_list(t, 3)
        self.assertEqual(res, val)

    def test_nest_list_2(self):
        t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -23]

        val = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, -23]]
        res = ut.nest_list(t, 3)
        self.assertEqual(res, val)   

    def test_nest_list_3(self):
        t = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -23]

        val = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -23]]
        res = ut.nest_list(t, 12)
        self.assertEqual(res, val)        


    def test_flatten_label_coordinates(self):

        c = {100 : [(1, 2, 3), (4, 5, 6), (7, 8, 9)], 200 : [(10, 11, 12), (13, 14, 15)]}

        val = [(100, (1, 2, 3)), (100, (4, 5, 6)), (100, (7, 8, 9)), \
                 (200, (10, 11, 12)), (200, (13, 14, 15))]

        print(val)


        res = ut.flatten_label_coordinates(c)
        res.sort()
        print(res)
        self.assertEqual(res, val)         

    def test_get_mx_pos_for_tpl(self):

        size = (3, 3)
        shape = (10, 10)

        maxpos = np.array((7, 7)) 
        print(ut.get_max_pos_for_tpl(size, shape))
        self.assertTrue((maxpos==ut.get_max_pos_for_tpl(size, shape)).all())    

    def test_get_random_pos_for_coordinate(self):
        coor = (5, 4)
        size = (4, 3)
        shape = (100, 100)
        

        pos_allowed = [(2, 2), (3, 2), (4, 2), (5, 2), \
                       (2, 3), (3, 3), (4, 3), (5, 3), \
                       (2, 4), (3, 4), (4, 4), (5, 4), \
                        ]

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertTrue(rpos in pos_allowed)


    def test_get_random_pos_for_coordinate_2(self):
        coor = (5, 4)
        size = (1, 1)
        shape = (100, 100)

        

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertEqual(rpos, (5, 4))   




    def test_get_random_pos_for_coordinate_3(self):
        coor = (5, 4)
        size = (4, 3)
        shape = (7, 7)
        

        pos_allowed = [(2, 2), (3, 2), \
                       (2, 3), (3, 3), \
                       (2, 4), (3, 4)\
                        ]

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertTrue(rpos in pos_allowed) 



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

    def test_add_to_filename(self):

        path = 'path/to/tiff/file.tif'
        add_str = 'label_1'

        out_s = ut.add_to_filename(path, add_str, suffix=True)
        out_p = ut.add_to_filename(path, add_str, suffix=False)

        self.assertEqual(out_s, 'path/to/tiff/file_label_1.tif')
        self.assertEqual(out_p, 'path/to/tiff/label_1_file.tif')

    def test_add_to_filename_2(self):

        path = 'tifffile.tif'
        add_str = 'label_1'

        out_s = ut.add_to_filename(path, add_str, suffix=True)
        out_p = ut.add_to_filename(path, add_str, suffix=False)

        self.assertEqual(out_s, 'tifffile_label_1.tif')
        self.assertEqual(out_p, 'label_1_tifffile.tif')
    



    def test_remove_exclusive_vals_from_set(self):
        list_of_sets = [{1, 2, 3}, {1, 2}, {1, 4}, {5, 6}]

        res = ut.remove_exclusive_vals_from_set(list_of_sets)
        val = [{1, 2}, {1, 2}, {1}, set()]
        self.assertEqual(res, val)

    def test_remove_exclusive_vals_from_set(self):
        list_of_sets = [{1, 2, 3}, {1, 2}, {1, 4}, {5, 6}]

        res = ut.remove_exclusive_vals_from_set(list_of_sets)
        val = [{1, 2}, {1, 2}, {1}, set()]
        self.assertEqual(res, val)
    
    def test_get_exclusive_vals_from_set(self):
        list_of_sets = [{1, 2, 3}, {1, 2}, {1, 4}, {5, 6}]

        res = ut.get_exclusive_vals_from_set(list_of_sets)
        val = [{3}, set(), {4}, {5, 6}]
        self.assertEqual(res, val)

            
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
    
    def test_string_distance(self):
        a = ut.string_distance('hallo', 'hallo')
        print (a)
        b = ut.string_distance('hallo', 'hillu')
        print (b)


        self.assertEqual(0., a)
        self.assertEqual(.4, b)

    def test_compute_str_dist_matrix(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']

        mat, a, b = ut.compute_str_dist_matrix(a, b)
        print(mat)
        print(a)
        print(b)
        

    def test_compute_str_dist_matrix(self):

        a = ['hund', 'katze', 'maus']
        b = ['hund', 'mauser', 'kater', 'pferd']

        val = [['hund', 'hund'], ['katze', 'kater'], ['maus', 'mauser']]
        pairs = ut.find_best_matching_pairs(a, b)
        print(pairs)
        self.assertEqual(pairs, val)


    def test_compute_str_dist_matrix_2(self):

        a = ['img_6width4height3slices_rgb.tif', 'img_40width26height3slices_rgb.tif', 'img_40width26height6slices_rgb.tif']
        b = ['lbl_6width4height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif']

        val = [['img_6width4height3slices_rgb.tif', 'lbl_6width4height3slices_rgb.tif']\
            , ['img_40width26height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif']\
            , ['img_40width26height6slices_rgb.tif', None]]
            
        pairs = ut.find_best_matching_pairs(a, b)
        print(pairs)
        self.assertEqual(pairs, val)        


            
