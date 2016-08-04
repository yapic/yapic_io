from unittest import TestCase
import os


import yapic_io.utils as ut


class TestUtils(TestCase):
    def test_flatten_label_coordinates(self):

        c = {100 : [(1,2,3), (4,5,6), (7,8,9)], 200 : [(10, 11, 12), (13, 14, 15)]}

        val = [(100,(1,2,3)), (100, (4,5,6)), (100, (7,8,9)),\
                 (200, (10,11,12)), (200, (13,14,15))]

        print(val)


        res = ut.flatten_label_coordinates(c)
        res.sort()
        print(res)
        self.assertEqual(res, val)         

    def test_get_mx_pos_for_tpl(self):

        size = (3,3)
        shape = (10,10)

        maxpos = (7,7) 

        self.assertEqual(maxpos, ut.get_max_pos_for_tpl(size, shape))    

    def test_get_random_pos_for_coordinate(self):
        coor = (5,4)
        size = (4,3)
        shape = (100,100)
        

        pos_allowed = [(2,2), (3,2), (4,2), (5,2),\
                       (2,3), (3,3), (4,3), (5,3),\
                       (2,4), (3,4), (4,4), (5,4),\
                        ]

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertTrue(rpos in pos_allowed)


    def test_get_random_pos_for_coordinate_2(self):
        coor = (5,4)
        size = (1,1)
        shape = (100,100)

        

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertEqual(rpos,(5,4))   




    def test_get_random_pos_for_coordinate_3(self):
        coor = (5,4)
        size = (4,3)
        shape = (7,7)
        

        pos_allowed = [(2,2), (3,2),\
                       (2,3), (3,3),\
                       (2,4), (3,4)\
                        ]

        for i in list(range(1500)):                
            rpos = ut.get_random_pos_for_coordinate(coor, size, shape)
            self.assertTrue(rpos in pos_allowed) 



    def test_compute_pos(self):
        shape = (8,5)
        size = (5,2)

        res = ut.compute_pos(shape, size)
        val = [(0, 0), (0, 2), (0, 3), (3, 0), (3, 2), (3, 3)]
        self.assertEqual(res, val)      



    def test_compute_pos_2(self):
        shape = (6,4)
        size = (2,2)

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
    





            
