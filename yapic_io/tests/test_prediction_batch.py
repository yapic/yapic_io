import tempfile
from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_batch import PredictionBatch
import numpy as np
import yapic_io.utils as ut
base_path = os.path.dirname(__file__)
from yapic_io.factories import make_tiff_interface

class TestPredictionBatch(TestCase):
    def test_computepos_1(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)
        
        d = Dataset(c)

        size = (1,1,1)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(len(p._tpl_pos_all),6*4*3)
        self.assertEqual(len(p._tpl_pos_all),len(set(p._tpl_pos_all)))
        #self.assertTrue(False)

    def test_computepos_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (3,6,4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0))])


    def test_getr_batch_index_list(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        
        

        d = Dataset(c)

        size = (2,6,4)
        batch_size = 10

        p = PredictionBatch(d, batch_size, size)

        print(len(p._tpl_pos_all))
        res = p._batch_index_list
        print(res)
        self.assertEqual(len(res[0]),batch_size)
        self.assertEqual(len(res[-1]),7)


    def test_computepos_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (2,6,4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._tpl_pos_all)
        print(len(p._tpl_pos_all))
        print(sorted(list(set(p._tpl_pos_all))))
        self.assertEqual(p._tpl_pos_all, [(0,(0,0,0)), (0,(1,0,0))])    


    def test_getitem(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        #batch size is 2, so the first 2 templates go with the first batch
        #(size two), the third template in in the second batch. the second
        #batch has only size 1 (is smaller than the specified batch size),
        #because it contains the rest. 

        print(p[0].pixels().shape)
        print(p[1].pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0].pixels().shape, (2,3,1,6,4))
        self.assertEqual(p[1].pixels().shape, (1,3,1,6,4))
       


    def test_getitem(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 3

        p = PredictionBatch(d, batch_size, size)
        #batch size is 3, this means all3 tempplates fit in one batch

        print(p[0].pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].pixels().shape, (3,3,1,6,4))
        

    def test_get_actual_batch_size(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)


        self.assertEqual(p[0].get_actual_batch_size(), 2)
        self.assertEqual(p[1].get_actual_batch_size(), 1)

    def test_get_curr_tpl_indices(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)


        self.assertEqual(p[0].get_curr_tpl_indices(), [0,1])
        self.assertEqual(p[1].get_curr_tpl_indices(), [2])    


    def test_get_curr_tpl_positions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path,label_path)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)


        self.assertEqual(p[0].get_curr_tpl_positions(), [(0,(0,0,0)), (0,(1,0,0))])
        self.assertEqual(p[1].get_curr_tpl_positions(), [(0,(2,0,0))])        
        
    def test_put_probmap_data(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        
        d = Dataset(c)

        size = (1,6,4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((1,2,1,6,4))
        p[0].put_probmap_data(data)
        p[1].put_probmap_data(data)
        p[2].put_probmap_data(data)

    def test_put_probmap_data_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)

        d = Dataset(c)

        size = (1,6,4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2,2,1,6,4))
        p[0].put_probmap_data(data)

        data = np.ones((1,2,1,6,4))
        p[1].put_probmap_data(data)


    def test_put_probmap_data_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        d = Dataset(c)

        size = (1,3,4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2,3,1,3,4))
        p[0].put_probmap_data(data)

        data = np.ones((2,3,1,3,4))
        p[1].put_probmap_data(data)  

        data = np.ones((2,3,1,3,4))
        p[2].put_probmap_data(data)       
    


    def test_put_probmap_data_multichannel_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        d = Dataset(c)

        print('labels')
        print(c.labelvalue_mapping)
        c.load_label_filenames('*')    
        c.map_labelvalues()
        print('labels')
        print(c.labelvalue_mapping)
        d = Dataset(c)
        
        size = (1,3,4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)
        print('_labels')
        print(p._labels)

        data = np.ones((1,6,1,3,4))
        p[0].put_probmap_data(data)

    
    def test_prediction_loop(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            #mock classification function
            def classify(pixels, value):
                return np.ones(pixels.shape) * value

            #define data loacations
            pixel_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
            label_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')
            savepath = tmpdirname

            tpl_size = (1,5,4) # size of network output layer in zxy
            padding = (0,0,0) # padding of network input layer in zxy, in respect to output layer

             # make training_batch mb and prediction interface p with TiffConnector binding
            _, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size, padding_zxy=padding, training_batch_size=2) 

            self.assertEqual(len(p), 255)
            self.assertEqual(p.get_labels(), [1,2,3])

            #classify the whole bound dataset
            for counter, item in enumerate(p):
                pixels = item.pixels() #input for classifier
                mock_classifier_result = classify(pixels, counter) #classifier output
                #pass classifier results for each class to data source
                item.put_probmap_data(mock_classifier_result)     


    def test_put_probmap_data_for_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*')
        savepath = os.path.join(base_path, '../test_data/tmp/')
        c = TiffConnector(img_path,label_path, savepath=savepath)
        d = Dataset(c)

        d = Dataset(c)

        size = (3,3,3)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((3,3,3))
        
        path1 = savepath + '40width26height3slices_rgb_class_109.tif'
        path2 = savepath + '40width26height6slices_rgb_class_109.tif'
        path3 = savepath + '6width4height3slices_rgb_class_109.tif'
        
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

        print(len(p))
        print(p._tpl_pos_all)
        p._put_probmap_data_for_label(data, label=1, tpl_pos_index=0) #first
        p._put_probmap_data_for_label(data, label=1, tpl_pos_index=381) #last

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

