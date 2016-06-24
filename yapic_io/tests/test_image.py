import os
from unittest import TestCase

import numpy as np
import yapic_io.image as im
import yapic_io.image_importers as ip

class TestImage(TestCase):
    def test_get_template_value_error_size(self):
        image = np.zeros((4, 5, 6)) #3 dim image
        pos = (0, 0)
        size = (3, 2)

        self.assertRaises(ValueError\
            , lambda: im.get_template(image, pos, size))  
        

    def test_get_template_value_error_pos(self):
        image = np.zeros((4, 5, 6)) #3 dim image
        pos = (0, 0)
        size = (3, 2, 2)
        
        self.assertRaises(ValueError\
            , lambda: im.get_template(image, pos, size))  


    def test_are_all_elements_uneven(self):
        a = (5, 7, 11)
        b = (5, 7, 12)

        self.assertTrue(im.are_all_elements_uneven(a))
        self.assertFalse(im.are_all_elements_uneven(b))


    def test_get_template(self):
        image = np.zeros((5, 7)).astype(int) #3 dim image
        
        image = np.array([[0, 0, 0, 0, 0]\
                        , [0, 0, 1, 2, 0]\
                        , [0, 0, 3, 4, 0]\
                        , [0, 0, 5, 6, 0]])

        pos = (1,2)
        size = (3, 2)

        val = np.array([[1,2]\
                       ,[3,4]\
                       ,[5,6]])
        res = im.get_template(image, pos, size)
        print(res)
        self.assertTrue((res == val).all())
        

    def test_get_template_err_outofbounds(self):
        image = np.zeros((5, 7)).astype(int) #3 dim image
        
        image = np.array([[0, 0, 0, 0, 0]\
                        , [0, 0, 1, 2, 0]\
                        , [0, 0, 3, 4, 0]\
                        , [0, 0, 5, 6, 0]])

        pos = (1,2)
        size = (4, 2)

        self.assertRaises(ValueError\
            , lambda: im.get_template(image, pos, size))  


    def test_get_indices_error_dims(self):
        pos = (1,2,3)
        size = (1,2)

        self.assertRaises(ValueError\
            , lambda: im.get_indices(pos, size))  


    def test_get_indices(self):
        
        pos = [5, 5]
        size = (3, 5)

        validation_ind = np.array([[5, 6, 7], [5, 6, 7, 8, 9]])
        ind = im.get_indices(pos, size)
        
        self.assertTrue((ind == validation_ind).all())


    def test_get_padding_size_1(self):

        shape = (7, 11)
        pos = (5, 5)
        size = (3, 5)
        res = im.get_padding_size(shape, pos, size)
        self.assertEqual([(0, 1), (0, 0)], res)


    def test_get_padding_size_2(self):

        shape = (7, 11)
        pos = (-3, 0)
        size = (3, 5)

        res = im.get_padding_size(shape, pos, size)
        self.assertEqual([(3,0),(0,0)], res) 


    def test_correct_pos_for_padding(self):
        pos = (-1,-1)
        padding_sizes = ((1,0), (1,0))
        res = im.correct_pos_for_padding(pos, padding_sizes)
        val = (0,0)
        self.assertEqual(val, res)       


    def test_get_template_withpadding(self):
        image = np.array([[0, 1, 2, 3, 4]\
                         ,[5, 6, 7, 8, 9]\
                         ,[10,11,12,13,14]\
                         ,[15,16,17,18,19]\
                         ,[20,21,22,23,24]\
                        ])       
        
        tpl_val = np.array([[6, 5, 6, 7, 8, 9]\
                           ,[1, 0, 1, 2, 3, 4]\
                           ,[6, 5, 6, 7, 8, 9]\
                           ,[11,10,11,12,13,14]\
                           ,[16,15,16,17,18,19]\
                           ,[21,20,21,22,23,24]\
                           ,[16,15,16,17,18,19]\
                           ,[11,10,11,12,13,14]\
                           ])       

        pos = (1,1)
        size = (4,2)
        padding = 2

        tpl = im.get_template(image, pos, size, padding=2)

        print(tpl)

        self.assertTrue((tpl == tpl_val).all())


    def test_init_image(self):
        path = os.path.join(os.path.dirname(__file__), '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')

        img = im.Image(ip.import_tiff_image, [path])

        print(img)
        self.assertEqual(img.dims, (3, 2, 6, 4))



    def test_get_template_meshgrid(self):
        image_shape = (5, 4)
        pos = (1, 2)
        size = (3, 2)
        m = im.get_template_meshgrid(image_shape, pos, size)
        print(m)
        vals_x = np.array([[1,1], [2,2], [3,3]])
        vals_y = np.array([[2, 3], [2, 3], [2, 3]])
        
        self.assertTrue((m[0] == vals_x).all())
        self.assertTrue((m[1] == vals_y).all())

