from unittest import TestCase
import os
from yapic_io.tiffconnector import Tiffconnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_data import Prediction_data

import yapic_io.utils as ut
base_path = os.path.dirname(__file__)

class TestPredictiondata(TestCase):
    def test_computepos_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (1,1,1)

        p = Prediction_data(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(len(p._tpl_pos_all),6*4*3)
        self.assertEqual(len(p._tpl_pos_all),len(set(p._tpl_pos_all)))
        #self.assertTrue(False)

    def test_computepos_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (3,6,4)

        p = Prediction_data(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0))])


    def test_computepos_3(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (2,6,4)

        p = Prediction_data(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0)), (0,(1,0,0))])    


    def test_getitem(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (1,6,4)

        p = Prediction_data(d, size)

        print(p[0].get_pixels().shape)
        print(p[1].get_pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 3)
        self.assertEqual(p[0].get_pixels().shape, (3,1,6,4))
        self.assertEqual(p[1].get_pixels().shape, (3,1,6,4))
        self.assertEqual(p[2].get_pixels().shape, (3,1,6,4))

            

