from unittest import TestCase
import os
import numpy as np

from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.utils import get_template_meshgrid
import yapic_io.dataset as ds
import yapic_io.image_importers as ip
from pprint import pprint
import logging
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestDataset(TestCase):
    def test_n_images(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path,'path/to/nowhere/')
        d = Dataset(c)

        self.assertEqual(d.n_images, 3)
        

        
        

        
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

    def test_get_padding_size_3(self):

        shape = (7, 11, 7)
        pos = (5, 5, 5)
        size = (3, 5, 3)
        res = ds.get_padding_size(shape, pos, size)
        self.assertEqual([(0, 1), (0, 0), (0, 1)], res)

    def test_get_padding_size_3(self):

        shape = (7, 11, 7, 1)
        pos = (5, 5, 5, 0)
        size = (3, 5, 3, 1)
        res = ds.get_padding_size(shape, pos, size)
        self.assertEqual([(0, 1), (0, 0), (0, 1), (0, 0)], res)    

        

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



    def test_calc_inner_template_size_1(self):

        shape = (7, 11)
        pos = (-4, 8)
        size = (16, 9)

        pos_val = (0, 5)
        size_val = (7, 6)
        pos_tpl_val = (0, 3)
        pos_out, size_out, pos_tpl, padding = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)    
        self.assertEqual(pos_tpl_val, pos_tpl)  
        self.assertEqual(padding, [(4,5),(0,6)])

    def test_calc_inner_template_size_2(self):

        shape = (7, 11)
        pos = (-4, -1)
        size = (5, 8)

        pos_val = (0, 0)
        size_val = (4, 7)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,0))
        self.assertEqual(pd, [(4,0),(1,0)])        

    def test_calc_inner_template_size_3(self):

        shape = (7, 11)
        pos = (-2, -3)
        size = (3, 4)

        pos_val = (0, 0)
        size_val = (2, 3)
        
        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,0))
        self.assertEqual(pd, [(2,0),(3,0)])        

    def test_calc_inner_template_size_4(self):

        shape = (7, 11)
        pos = (-5, 2)
        size = (6, 4)

        pos_val = (0, 2)
        size_val = (5, 4)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,0))
        self.assertEqual(pd, [(5,0),(0,0)])        


    def test_calc_inner_template_size_5(self):

        shape = (7, 11)
        pos = (2, -4)
        size = (3, 8)

        pos_val = (2, 0)
        size_val = (3, 4)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,0))
        self.assertEqual(pd, [(0,0),(4,0)])        
    def test_calc_inner_template_size_6(self):

        shape = (7, 11)
        pos = (2, 9)
        size = (3, 6)

        pos_val = (2, 7)
        size_val = (3, 4)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,2))
        self.assertEqual(pd, [(0,0),(0,4)]) 

    def test_calc_inner_template_size_7(self):

        shape = (7, 11)
        pos = (2, 9)
        size = (6, 5)

        pos_val = (2, 8)
        size_val = (5, 3)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,1))
        self.assertEqual(pd, [(0,1),(0,3)]) 

    def test_calc_inner_template_size_8(self):

        shape = (7, 11)
        pos = (5, 9)
        size = (5, 6)

        pos_val = (4, 7)
        size_val = (3, 4)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(tuple(pos_out), pos_val)
        self.assertEqual(tuple(size_out), size_val)        
        self.assertEqual(tuple(pos_tpl), (1,2))
        self.assertEqual(pd, [(0,3),(0,4)]) 


    def test_calc_inner_template_size_9(self):

        shape = (7, 11)
        pos = (-4, -1)
        size = (6, 6)

        pos_val = (0, 0)
        size_val = (4, 5)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(tuple(pos_out), pos_val)
        self.assertEqual(tuple(size_out), size_val)        
        self.assertEqual(tuple(pos_tpl), (0,0))
        self.assertEqual(pd, [(4,0),(1,0)]) 

    def test_calc_inner_template_size_10(self):

        shape = (7, 11)
        pos = (6, 7)
        size = (4, 3)

        pos_val = (4, 7)
        size_val = (3, 3)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (2,0))
        self.assertEqual(pd, [(0,3),(0,0)]) 



    def test_calc_inner_template_size_11(self):

        shape = (7, 11)
        pos = (2, 2)
        size = (3, 4)

        pos_val = (2, 2)
        size_val = (3, 4)

        pos_out, size_out, pos_tpl, pd = \
                ds.calc_inner_template_size(shape, pos, size)

        print(pos_out)
        print(size_out)

        self.assertEqual(pos_out, pos_val)
        self.assertEqual(size_out, size_val)        
        self.assertEqual(pos_tpl, (0,0))
        self.assertEqual(pd, [(0,0),(0,0)])                 
                      
      
    def test_is_padding(self):
        self.assertTrue(ds.is_padding([(2,3),(0,0)]))
        self.assertTrue(ds.is_padding([(2,3),(20,3)]))
        self.assertTrue(ds.is_padding([(0,0),(3,0)]))
        self.assertFalse(ds.is_padding([(0,0),(0,0)]))

    def test_get_template_singlechannel_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path,'path/to/nowhere/')
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif', None)\
                , ('40width26height3slices_rgb.tif', None)\
                , ('40width26height6slices_rgb.tif', None)\
                ]
        
        d = Dataset(c)
        image_nr = 0
        pos = (0,0,0)
        size= (3, 6, 4)
        channel = 0
        tpl = d.get_template_singlechannel(image_nr, pos, size, channel, reflect=False)

        
        self.assertEqual(tpl.shape,(3,6,4))


    def test_get_template_singlechannel_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path,'path/to/nowhere/')
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif', None)\
                , ('40width26height3slices_rgb.tif', None)\
                , ('40width26height6slices_rgb.tif', None)\
                ]
        
        d = Dataset(c)
        image_nr = 0
        pos = (0,0,0)
        size= (1, 6, 4)
        channel = 0
        tpl = d.get_template_singlechannel(image_nr, pos, size, channel, reflect=False)

        val = \
        [[[151, 132, 154, 190],\
           [140,  93, 122, 183],\
           [148, 120, 133, 171],\
           [165, 175, 166, 161],\
           [175, 200, 184, 161],\
           [174, 180, 168, 157]]]

        val = np.array(val) 
        print(val)
        print(tpl)      
        self.assertTrue((tpl == val).all())   


    def test_get_template_singlechannel_3(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path,'path/to/nowhere/')
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif', None)\
                , ('40width26height3slices_rgb.tif', None)\
                , ('40width26height6slices_rgb.tif', None)\
                ]
        
        d = Dataset(c)
        image_nr = 0
        pos = (0,-1,-2)
        size= (1, 3, 3)
        channel = 0
        tpl = d.get_template_singlechannel(image_nr, pos, size, channel, reflect=True)

        val = \
        [[[132, 151, 151],\
           [132,  151, 151],\
           [93, 140, 140]]]
        
        val = np.array(val) 
        print(val)
        print(tpl)   
        self.assertTrue((tpl == val).all())



    def test_get_template_singlechannel_4(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = TiffConnector(img_path,'path/to/nowhere/')
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif',  None)\
                , ('40width26height3slices_rgb.tif', None)\
                , ('40width26height6slices_rgb.tif', None)\
                ]
        
        d = Dataset(c)
        image_nr = 0
        pos = (0, 4, 2)
        size= (1, 2, 6)
        channel = 0
        tpl = d.get_template_singlechannel(image_nr, pos, size, channel, reflect=True)

        val = \
        [[[184, 161, 161, 184, 200, 175],\
           [168,  157, 157, 168, 180, 174]]]
        
        val = np.array(val) 
        print(val)
        print(tpl)   
        self.assertTrue((tpl == val).all())


    def test_label_coordinates_to_5d(self):

        label_dict = \
            {
            2 : np.array([(1, 3, 0, 0), (1, 5, 1, 2)]),
            3 : np.array([(1, 3, 1, 2), (1, 5, 3, 4), (1,1,1,1)]),
            }

        image_nr = 2

        val = \
            {
            2 : np.array([(2, 1, 3, 0, 0), (2, 1, 5, 1, 2)]),
            3 : np.array([(2, 1, 3, 1, 2), (2, 1, 5, 3, 4), (2, 1,1,1,1)]),
            }
        
        out = ds.label_coordinates_to_5d(label_dict, image_nr)
        print(val)
        print(out)
        self.assertTrue((val[2]==out[2]).all())
        self.assertTrue((val[3]==out[3]).all())
      
    

    def test_label_coordinates_is_valid(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif', None)\
                , ('40width26height3slices_rgb.tif', None)\
                , ('40width26height6slices_rgb.tif', None)\
                ]
        
        d = Dataset(c)

        #duplicates within one label are not allowed
        coor = \
            {
            2 : np.array([(2, 0, 3, 0, 0), (2, 0, 5, 1, 2), (2, 0, 3, 0, 0)]),
            3 : np.array([(2, 0, 3, 1, 2), (2, 0, 5, 3, 4), (2, 0,1,1,1), (2, 0, 3, 0, 0)]),
            }
        self.assertFalse(d.label_coordinates_is_valid(coor))    
        

        
        #duplicates for different labels are allowed
        #duplicate
        coor = \
            {
            2 : np.array([(2, 0, 3, 0, 0), (2, 0, 5, 1, 2)]),
            3 : np.array([(2, 0, 3, 1, 2), (2, 0, 5, 3, 4), (2, 0,1,1,1), (2, 0, 3, 0, 0)]),
            }
        
        print(coor)
        self.assertTrue(d.label_coordinates_is_valid(coor))       



        # self.assertFalse(d.label_coordinates_is_valid(coor))

        # one tuple with 4 instead 5 coordinate dimensions
        coor = \
            {
            2 : np.array([(2, 0, 3, 0), (2, 0, 5, 1)]),
            3 : np.array([(2, 0, 3, 1), (2, 0, 5, 3), (2, 0,1,1)]),
            }


        self.assertFalse(d.label_coordinates_is_valid(coor))  

        # image nr too high
        coor = \
            {
            2 : np.array([(2, 0, 3, 0, 0), (2, 0, 5, 1, 2)]),
            3 : np.array([(2, 0, 3, 1, 2), (3, 0,1,1,1)]),
            }


        self.assertFalse(d.label_coordinates_is_valid(coor))     


        # out of image bounds
        coor = \
            {
            2 : np.array([(0, 0, 3, 7, 4), (0, 0, 3, 1, 2)])
            }

        self.assertFalse(d.label_coordinates_is_valid(coor))

        # out of image bounds
        coor = \
            {
            2 : np.array([(0, 0, -1, 6, 4), (0, 0, 3, 1, 2)])
            }

        self.assertFalse(d.label_coordinates_is_valid(coor))

        # out of image bounds
        coor = \
            {
            2 : np.array([(0, 2, 0, 6, 4), (0, 1, 3, 1, 2)])
            }

        self.assertFalse(d.label_coordinates_is_valid(coor))

        # correct
        coor = \
            {
            2 : np.array([(0, 0, 0, 6, 4), (0, 0, 3, 1, 2)])
            }

        self.assertTrue(d.label_coordinates_is_valid(coor))         
  


    
    
    def test_get_weight_template_for_label(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        d.load_label_coordinates()

        img_nr = 0
        pos_czxy = (0,0,0)
        size_czxy = (1,6,4)
        label_value = 3
        mat = d._get_template_for_label_inner(img_nr,\
         pos_czxy, size_czxy, label_value)
        
        val = np.zeros(size_czxy)
        val[0,0,1] = 1
        val[0,4,1] = 1
        val[0,5,1] = 1 

        self.assertTrue((val==mat).all())


    def test_get_template_for_label_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        d.load_label_coordinates()

        img_nr = 0
        pos_czxy = np.array((0,0,0))
        size_czxy = np.array((1,6,4))
        label_value = 3
        mat = d.get_template_for_label(img_nr,\
         pos_czxy, size_czxy, label_value)
        
        val = np.zeros(size_czxy)
        val[0,0,1] = 1
        val[0,4,1] = 1
        val[0,5,1] = 1 

        self.assertTrue((val==mat).all())   
        self.assertEqual(len(mat.shape),3)

    def test_get_template_for_label_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        d.load_label_coordinates()

        img_nr = 0
        pos_czxy = np.array((0,2,1))
        size_czxy = np.array((1,7,2))
        label_value = 3
        mat = d.get_template_for_label(img_nr,\
         pos_czxy, size_czxy, label_value)
        
        val = np.zeros(size_czxy)
        # val[0,0,1] = 1
        # val[0,4,1] = 1
        # val[0,5,1] = 1 
        print(d.label_coordinates[3])
        print('tpl')
        print(mat)

        val = [\
            [0.,0.],\
            [0.,0.],\
            [1.,0.],\
            [1.,0.],\
            [1.,0.],\
            [1.,0.],\
            [0.,0.]\
        ]
        val = np.array(val)
        self.assertTrue((mat==val).all())         


    def test_get_weight_template_for_label_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        d.load_label_coordinates()
        d.set_weight_for_label(1.2, 3)
        img_nr = 0
        pos_czxy = np.array((0,0,0))
        size_czxy = np.array((1,6,4))
        label_value = 3
        pprint(d.label_coordinates)
        mat = d._get_template_for_label_inner(img_nr,\
         pos_czxy, size_czxy, label_value)
        
        val = np.zeros(size_czxy)
        print('valshae')
        print(val.shape)
        val[0,0,1] = 1.2
        val[0,4,1] = 1.2
        val[0,5,1] = 1.2 
        pprint('mat')
        pprint(mat)
        pprint(val)
        self.assertTrue((val==mat).all())    



    def test_load_label_coordinates(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()        

        val = \
            {
                1: np.array([(2, 0, 0, 0, 0), (2, 0, 0, 3, 1), (2, 0, 0, 3, 2), (2, 0, 0, 4, 1)]),
                2: np.array([(0, 0, 0, 2, 1), (0, 0, 0, 2, 2), (0, 0, 0, 2, 3), (0, 0, 0, 3, 1), (0, 0, 0, 3, 2), (0, 0, 0, 3, 3), (0, 0, 1, 5, 0), (0, 0, 1, 5, 1), (0, 0, 2, 0, 0), (0, 0, 2, 1, 0), (0, 0, 2, 1, 2), (2, 0, 2, 9, 7), (2, 0, 2, 9, 8), (2, 0, 2, 12, 7)]),
                3: np.array([(0, 0, 0, 0, 1), (0, 0, 0, 4, 1), (0, 0, 0, 5, 1), (2, 0, 0, 2, 2), (2, 0, 1, 0, 2), (2, 0, 1, 39, 25)])
            }
        
        d = Dataset(c)
        d.load_label_coordinates()
        print(d.label_coordinates)
        #self.assertTrue(False)
        self.assertTrue((val[1] == d.label_coordinates[1]).all())
        self.assertTrue((val[2] == d.label_coordinates[2]).all())
        self.assertTrue((val[3] == d.label_coordinates[3]).all())
    
    def test_load_label_coordinates_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        d.load_label_coordinates()
        d.init_label_weights()
        print('label_weights')
        print(d.label_weights)
        print('label_coordinates')
        print(d.label_coordinates)
        #check if dimensions match
        self.assertEqual(d.label_weights.keys(), d.label_coordinates.keys())
        self.assertEqual(len(d.label_weights[1]), len(d.label_coordinates[1]))
        self.assertEqual(len(d.label_weights[2]), len(d.label_coordinates[2]))
        self.assertEqual(len(d.label_weights[3]), len(d.label_coordinates[3]))
        
        self.assertEqual(set(d.label_weights[1]), set([1]))    
        self.assertEqual(set(d.label_weights[2]), set([1]))    
        self.assertEqual(set(d.label_weights[3]), set([1]))    

    
    def test_set_weight_for_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        
        d.set_weight_for_label(0.7, 3)

        self.assertTrue((d.label_weights[3] == np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])).all())
        self.assertTrue((d.label_weights[1] == np.array([1, 1, 1, 1])).all())

    def test_equalize_label_weights(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)
        
        d.equalize_label_weights()

        print(d.label_weights)

        res = np.unique(np.array(d.label_weights[1])) + \
            np.unique(np.array(d.label_weights[2])) + \
            np.unique(np.array(d.label_weights[3])) 
        
        print(res)
        self.assertEqual(np.round(res[0],3), 1.)

            

    def test_label_to_mask(self):
        shape = (5,4)
        pos = (2,1)
        size = (3,2)
        label_coors = np.array([(0,0), (2,1), (2,2)])
        weights = np.array([2, 2, 2])
        mat = ds.label2mask(shape, pos, size, label_coors, weights)    
        val = \
            [[2., 2.],\
            [0., 0.],\
            [0., 0.]]

        val = np.array(val)    


        print(mat)
        self.assertTrue((val == mat).all())


    def test_weights_to_mask(self):
        shape = (5,4)
        pos = (2,1)
        size = (3,2)
        label_coors = np.array([(0,0), (2,1), (2,2)])
        weights = np.array([1, 2, 3])
        
        mat = ds.label2mask(shape, pos, size, label_coors, weights)    
        val = \
            [[2., 3.],\
            [0., 0.],\
            [0., 0.]]

        val = np.array(val)    


        print(mat)
        self.assertTrue((val == mat).all())    

    def test_calc_equalized_label_weights(self):
        
        label_n = {1: 10, 2 : 20}

        weights = ds.calc_equalized_label_weights(label_n)

        print(weights)
        self.assertEqual(weights[1]/2, weights[2])
        self.assertEqual(weights[1]+weights[2], 1)

    def test_get_augmented_template(self):
        
        

        '''
        im = 
        [[ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  3.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]]


         template without augmentation
         tpl = 
          [[ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 1.  1.  1.  1.  3.  1.  1.  1.  1.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]
           [ 0.  0.  0.  0.  2.  0.  0.  0.  0.]]

        augmented template afer 45 degree rotation
        tpl_rot =
            [[ 1.  1.  0.  0.  0.  0.  0.  0.  2.]
             [ 0.  1.  1.  0.  0.  0.  0.  2.  2.]
             [ 0.  0.  1.  1.  0.  0.  2.  2.  0.]
             [ 0.  0.  0.  1.  1.  2.  2.  0.  0.]
             [ 0.  0.  0.  0.  3.  3.  0.  0.  0.]
             [ 0.  0.  0.  2.  2.  1.  1.  0.  0.]
             [ 0.  0.  2.  2.  0.  0.  1.  1.  0.]
             [ 0.  2.  2.  0.  0.  0.  0.  1.  1.]
             [ 2.  2.  0.  0.  0.  0.  0.  0.  1.]]

        '''

        def get_tpl_func(pos=None, size=None, img=None):
            return img[get_template_meshgrid(img.shape, pos, size)]


        val = np.array(\
            [[[[1., 1., 0., 0., 0., 0., 0., 0., 2.],\
             [0., 1., 1., 0., 0., 0., 0., 2., 2.],\
             [0., 0., 1., 1., 0., 0., 2., 2., 0.],\
             [0., 0., 0., 1., 1., 2., 2., 0., 0.],\
             [0., 0., 0., 0., 3., 3., 0., 0., 0.],\
             [0., 0., 0., 2., 2., 1., 1., 0., 0.],\
             [0., 0., 2., 2., 0., 0., 1., 1., 0.],\
             [0., 2., 2., 0., 0., 0., 0., 1., 1.],\
             [2., 2., 0., 0., 0., 0., 0., 0., 1.]]]])



        im = np.zeros((1, 1, 15, 15))
        im[0,0, 7, :] = 1
        im[0,0, :, 7] = 2
        im[0,0, 7, 7] = 3

        pos = np.array((0,0,3,3))
        size=np.array((1,1,9,9))
        tpl = get_tpl_func(pos=pos, size=size, img=im)

        tpl_rot = ds.get_augmented_template(im.shape, pos, size, \
        get_tpl_func, rotation_angle=45, shear_angle=0, **{'img': im})

        print(im)
        print(tpl)
        print(tpl_rot)

        self.assertTrue((val==tpl_rot).all())



    def test_get_augmented_template_2(self):
    
    

        '''
        im = 
        [[ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 1.  1.  1.  1.  1.  1.  1.  3.  1.  1.  1.  1.  1.  1.  1.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]
         [ 0.  0.  0.  0.  0.  0.  0.  2.  0.  0.  0.  0.  0.  0.  0.]]

        '''

        def get_tpl_func(pos=None, size=None, img=None):
            return img[get_template_meshgrid(img.shape, pos, size)]


        val = np.array(\
            [[[[3.]]]])



        im = np.zeros((1, 1, 15, 15))
        im[0, 0, 7, :] = 1
        im[0, 0, :, 7] = 2
        im[0, 0, 7, 7] = 3

        pos =np.array((0,0,7,7))
        size=np.array((1,1,1,1))
        tpl = get_tpl_func(pos=pos, size=size, img=im)

        tpl_rot = ds.get_augmented_template(im.shape, pos, size, \
        get_tpl_func, rotation_angle=45, shear_angle=0, **{'img': im})

        print(im)
        print(tpl)
        print(tpl_rot)

        self.assertTrue((val==tpl_rot).all())


    
    def test_get_multichannel_pixel_template_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        img = 0
        pos_zxy = (0,0,0)
        size_zxy = (1,4,3)
        channels = [1]

        val = np.array(\
            [[[[102, 89, 82]\
               ,[ 81, 37, 43]\
               ,[ 87, 78, 68]\
               ,[107, 153,125]]]]\
            )

        tpl = d.get_multichannel_pixel_template(img, pos_zxy, size_zxy, channels)
        print(tpl.shape)
        print(tpl)
        print(val)
        self.assertTrue((tpl==val).all())



    def test_get_multichannel_pixel_template_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        img = 0
        pos_zxy = (0,0,0)
        size_zxy = (1,4,3)
        channels = [1]
        pd = (0,2,3)

        val = np.array(\
            [[[[43, 37,  81,  81, 37, 43, 78, 78, 43]\
              ,[82, 89, 102, 102, 89, 82, 87, 87, 82]\
              ,[82, 89, 102, 102, 89, 82, 87, 87, 82]\
              ,[43, 37, 81, 81, 37, 43, 78, 78, 43]\
              ,[68, 78, 87, 87, 78, 68, 73, 73, 68]\
              ,[125, 153, 107, 107, 153,125, 82, 82, 125]\
              ,[161, 180, 121, 121, 180,161, 106, 106, 161]\
              ,[147, 143, 111, 111, 143,147, 123, 123, 147]]]]\
            )

        tpl = d.get_multichannel_pixel_template(img, pos_zxy, size_zxy, channels\
                , pixel_padding=pd, rotation_angle=0)
        print(tpl.shape)
        print(val.shape)
        print(tpl)
        print(val)
        self.assertTrue((tpl==val).all())





    def test_get_multichannel_pixel_template_3(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        img = 0
        pos_zxy = (0,0,0)
        size_zxy = (1,4,3)
        channels = [1]
        pd = (0,2,3)

        val = np.array(\
            [[[[ 73,  73,  43,  89,  89, 102,  81,  87,  78]\
               ,[ 82,  68,  78,  37, 102, 102, 102,  37,  68]\
               ,[161, 153,  78,  87,  81, 102,  89,  82,  78]\
               ,[180, 180, 107,  87,  87,  37,  82,  87,  87]\
               ,[143, 121, 121, 107,  78,  68,  78,  87,  87]\
               ,[111, 111, 121, 180, 125,  73,  73,  78,  82]\
               ,[121, 111, 143, 161, 106,  82,  73,  68,  43]\
               ,[121, 180, 147, 123, 106, 106, 125,  68,  78]]]]
            )

        tpl = d.get_multichannel_pixel_template(img, pos_zxy, size_zxy, channels\
                , pixel_padding=pd, rotation_angle=45)
        print(tpl.shape)
        print(val.shape)
        print(tpl)
        print(val)
        self.assertTrue((tpl==val).all())


    def test_get_training_template_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        img = 0
        pos_zxy = (0,0,0)
        size_zxy = (1,4,3)
        channels = [1]
        labels = [2, 3]
        

        tr = d.get_training_template(img, pos_zxy, size_zxy, channels, labels,\
            pixel_padding=(0, 1, 2))

        print(tr)
        self.assertEqual(tr.pixels.shape, (1,1,6,7))
        self.assertEqual(tr.weights.shape, (2,1,4,3))

        #self.assertTrue(False)


    def test_pick_random_label_coordinate(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)


        res_1 = d.pick_random_label_coordinate(equalized=False)
        res_2 = d.pick_random_label_coordinate(equalized=False)
        print(res_1)
        print(res_2)
        #self.assertTrue(False) 

    def test_pick_random_label_coordinate_for_label(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                , ['40width26height6slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', '40width26height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)


        res_1 = d.pick_random_label_coordinate_for_label(3)
        #res_2 = d.pick_random_label_coordinate_for_label(equalized=False)
        #print(res_1)
        print(res_1)
        self.assertEqual(3, res_1[0]) 

    
    def test_pick_random_training_template(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        

        d = Dataset(c)

        size = (1,3,4)
        pad = (1,2,2)
        channels = [0,1,2]

        pixel_shape_val = (3,3,7,8)
        weight_shape_val = (3,1,3,4)
        #labels_val = [91, 109, 150]
        
        #mapping [{91: 1, 109: 2, 150: 3}]
        labels_val = [1, 2, 3]
        tpl = d.pick_random_training_template(\
            size, channels, pixel_padding=pad, rotation_angle=45)

        
        self.assertEqual(tpl.pixels.shape,pixel_shape_val)
        self.assertEqual(tpl.channels,channels)
        self.assertEqual(tpl.weights.shape,weight_shape_val)
        self.assertEqual(tpl.labels,labels_val)


    def test_pick_random_training_template_spec_label(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        

        d = Dataset(c)

        size = (1,1,1)
        pad = (0,2,2)
        channels = [0,1,2]

        
        labels_val = [1, 2, 3]
        
        tpl_1 = d.pick_random_training_template(\
            size, channels, pixel_padding=pad, rotation_angle=45, label_region=1)


        tpl_2 = d.pick_random_training_template(\
            size, channels, pixel_padding=pad, rotation_angle=45, label_region=2)

        tpl_3 = d.pick_random_training_template(\
            size, channels, pixel_padding=pad, rotation_angle=45, label_region=3)

        print(tpl_1[2])
        self.assertEqual(tpl_1[2][0][0][0][0],1)
        self.assertEqual(tpl_2[2][1][0][0][0],1)
        self.assertEqual(tpl_3[2][2][0][0][0],1)
        #self.assertEqual(tpl_1['weights'][0][0][0][0],1)

        # self.assertEqual(tpl.pixels.shape,pixel_shape_val)
        # self.assertEqual(tpl.channels,channels)
        # self.assertEqual(tpl.weights.shape,weight_shape_val)
        # self.assertEqual(tpl.labels,labels_val)
    

    
    def test_put_prediction_template_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        


        c = TiffConnector(img_path, label_path, savepath = savepath)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]
        c.load_label_filenames()        
        d = Dataset(c)
        #d.load_label_coordinates()
        #d.init_label_weights()


        pixels = np.array([[[.1, .2, .3],\
                            [.4, .5, .6]]], dtype=np.float32)
        
        path = savepath + '6width4height3slices_rgb_class_2.tif'
        
        try:
            os.remove(path)
        except:
            pass    

        d.put_prediction_template(pixels, pos_zxy=(0,1,1), image_nr=0, label_value=2)
        probim = ip.import_tiff_image(path, zstack=True)
        pprint(probim)

        val = \
        np.array([[[[ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.1       ,  0.2       ,  0.3       ],\
         [ 0.        ,  0.4       ,  0.5       ,  0.6       ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ]],\
        [[ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ]],\
        [[ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ],\
         [ 0.        ,  0.        ,  0.        ,  0.        ]]]], dtype=np.float32)
        
        self.assertTrue((val==probim).all())

        try:
            os.remove(path)
        except:
            pass    
    