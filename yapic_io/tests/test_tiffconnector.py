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
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]

        c.load_img_filenames()
        fnames = [e[0] for e in c.filenames]
        img_fnames = [e[0] for e in img_filenames]
        print('c filenames:')
        print(c.filenames)
        print('fnames')
        print(fnames)
        print(set(fnames))
        self.assertEqual(set(img_fnames), set(fnames))


    def test_load_filenames_emptyfolder(self):
        img_path = os.path.join(base_path, '../test_data/empty_folder/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        #c.load_img_filenames()
        self.assertIsNone(c.filenames)



    

    def test_load_img_dimensions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        c = Tiffconnector(img_path,'path/to/nowhere/')
        
        c.filenames = [\
                ('6width4height3slices_rgb.tif',)\
                , ('40width26height3slices_rgb.tif',)\
                , ('40width26height6slices_rgb.tif',)\
                ]


        
        self.assertFalse(c.load_img_dimensions(-1))
        self.assertFalse(c.load_img_dimensions(4))   
        
        self.assertEqual(c.load_img_dimensions(0), (3, 3, 6, 4))   
        self.assertEqual(c.load_img_dimensions(1), (3, 3, 40, 26))
        self.assertEqual(c.load_img_dimensions(2), (3, 6, 40, 26))

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


    def test_exists_label_for_image_nr(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]    
        c.load_label_filenames()        
        self.assertFalse(c.exists_label_for_img(2))
        self.assertTrue(c.exists_label_for_img(0))
        self.assertTrue(c.exists_label_for_img(1))

    def test_load_label_filenames(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]    
        print(c.filenames)
        c.load_label_filenames()
        print(c.filenames)
        self.assertIsNone(c.filenames[2][1])
        self.assertEqual(c.filenames[1][1], '40width26height3slices_rgb.tif')
        self.assertEqual(c.filenames[0][1], '6width4height3slices_rgb.tif')
        #self.assertTrue(False)
    

    def test_load_label_matrix(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]    
        print(c.filenames)
        c.load_label_filenames()
        im = c.load_image(0)
        labelmat = c.load_label_matrix(0)
        print(labelmat)
        self.assertEqual(labelmat.shape, (1, 3, 6, 4))

    def test_check_labelmat_dimensions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]    
        print(c.filenames)
        c.load_label_filenames()
        c.check_labelmat_dimensions()
        #self.assertTrue(False)


    def test_get_labelvalues_for_im(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]
        c.load_label_filenames()            
        labelvals = c.get_labelvalues_for_im(0)

        print('lbklals')
        print(labelvals)
        self.assertEqual(labelvals, [109, 150])

    def test_get_label_coordinates(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path, label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]
        c.load_label_filenames()            
        labelc = c.get_label_coordinates(0)
        labelmat = c.load_label_matrix(0)
        
        val_109 =\
           [(0, 0, 2, 1), (0, 0, 2, 2), (0, 0, 2, 3)\
           , (0, 0, 3, 1), (0, 0, 3, 2), (0, 0, 3, 3)\
           , (0, 1, 5, 0), (0, 1, 5, 1), (0, 2, 0, 0)\
           , (0, 2, 1, 0), (0, 2, 1, 2)]

        val_150 =\
            [(0, 0, 0, 1), (0, 0, 4, 1), (0, 0, 5, 1)]   

        print('109')
        print(labelc[109])
        print('150')
        print(labelc[150])

        print(labelmat)
        print(labelmat.shape)
        self.assertEqual(val_109, labelc[109])
        self.assertEqual(val_150, labelc[150])    
    
    

        


