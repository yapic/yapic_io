from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_data import PredictionData
import numpy as np
import yapic_io.utils as ut
base_path = os.path.dirname(__file__)

class TestPredictiondata(TestCase):
    def test_computepos_1(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (1,1,1)

        p = PredictionData(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(len(p._tpl_pos_all),6*4*3)
        self.assertEqual(len(p._tpl_pos_all),len(set(p._tpl_pos_all)))
        #self.assertTrue(False)

    def test_computepos_2(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (3,6,4)

        p = PredictionData(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0))])


    def test_computepos_3(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (2,6,4)

        p = PredictionData(d, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0)), (0,(1,0,0))])    


    def test_getitem(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        #c.load_label_filenames()    

        d = Dataset(c)

        size = (1,6,4)

        p = PredictionData(d, size)

        print(p[0].get_pixels().shape)
        print(p[1].get_pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 3)
        self.assertEqual(p[0].get_pixels().shape, (3,1,6,4))
        self.assertEqual(p[1].get_pixels().shape, (3,1,6,4))
        self.assertEqual(p[2].get_pixels().shape, (3,1,6,4))

    def test_put_probmap_data(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        size = (1,3,4)

        p = PredictionData(d, size)

        data = np.ones((2,1,3,4))
        p[0].put_probmap_data(data)


    def test_put_probmap_data_multichannel_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                ]
        print('labels')
        print(c.labelvalue_mapping)
        c.load_label_filenames()    
        c.map_labelvalues()
        print('labels')
        print(c.labelvalue_mapping)
        d = Dataset(c)
        print('label coors in dataset')
        print(d.label_coordinates.keys())
        size = (1,3,4)

        p = PredictionData(d, size)
        print('_labels')
        print(p._labels)

        data = np.ones((4,1,3,4))
        p[0].put_probmap_data(data)

        #self.assertTrue(False)

        
    def test_put_probmap_data_multichannel_label_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        
        
        print('labels')
        print(c.labelvalue_mapping)
        c.load_label_filenames()    
        c.map_labelvalues()
        print('labels')
        print(c.labelvalue_mapping)
        d = Dataset(c)
        print('label coors in dataset')
        print(d.label_coordinates.keys())
        size = (1,3,4)

        p = PredictionData(d, size)
        print('_labels')
        print(p._labels)

        data = np.ones((6,1,3,4))
        p[0].put_probmap_data(data)    


    def test_put_probmap_data_for_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        savepath = os.path.join(base_path, '../test_data/tmp/')



        c = TiffConnector(img_path,label_path, savepath = savepath)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', None]\
                , ['40width26height3slices_rgb.tif', None]\
                , ['40width26height6slices_rgb.tif', None]\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        size = (3,3,3)

        p = PredictionData(d, size)

        data = np.ones((3,3,3))
        
        path1 = savepath + '6width4height3slices_rgb_class_109.tif'
        path2 = savepath + '40width26height3slices_rgb_class_109.tif'
        path3 = savepath + '40width26height6slices_rgb_class_109.tif'
        
        try:
            os.remove(path1)
        except:
            pass
        try:
            os.remove(path2)
        except:
            pass
        try:
            os.remove(path3)
        except:
            pass               




        for t in p:
            data = data+1
            t.put_probmap_data_for_label(data, label=1)
    


        try:
            os.remove(path1)
        except:
            pass
        try:
            os.remove(path2)
        except:
            pass
        try:
            os.remove(path3)
        except:
            pass             

            
    # def test_is_tpl_size_valid(self):
    #     img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
    #     label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
    #     savepath = os.path.join(base_path, '../test_data/tmp/')



    #     c = TiffConnector(img_path,label_path, savepath = savepath)
        
    #     c.filenames = [\
    #             ['6width4height3slices_rgb.tif', None]\
    #             , ['40width26height3slices_rgb.tif', None]\
    #             , ['40width26height6slices_rgb.tif', None]\
    #             ]
    #     c.load_label_filenames()    

    #     d = Dataset(c)

    #     size = (3,3,3)
    #     p = PredictionData(d, size)
    #     self.assertTrue(p._is_tpl_size_valid(size))

    #     size = (3,6,4)
    #     p = PredictionData(d, size)
    #     self.assertTrue(p._is_tpl_size_valid(size))

    #     size = (3,6,5)
    #     p = PredictionData(d, size)
    #     self.assertFalse(p._is_tpl_size_valid(size))






