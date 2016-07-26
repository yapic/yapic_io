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