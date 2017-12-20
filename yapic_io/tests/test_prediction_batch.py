import tempfile
from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_batch import PredictionBatch
import numpy as np
import yapic_io.utils as ut
base_path = os.path.dirname(__file__)
from yapic_io import TiffConnector, Dataset, PredictionBatch

class TestPredictionBatch(TestCase):
    def test_computepos_1(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)
        
        d = Dataset(c)

        size = (1, 1, 1)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._all_tile_positions)
        print(len(p._all_tile_positions))
        print(sorted(list(set(p._all_tile_positions))))
        self.assertEqual(len(p._all_tile_positions), 6*4*3)
        self.assertEqual(len(p._all_tile_positions), len(set(p._all_tile_positions)))
        #self.assertTrue(False)

    def test_computepos_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (3, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._all_tile_positions)
        print(len(p._all_tile_positions))
        print(sorted(list(set(p._all_tile_positions))))
        self.assertEqual(p._all_tile_positions, [(0, (0, 0, 0))])


    def test_computepos_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (2, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        print(p._all_tile_positions)
        print(len(p._all_tile_positions))
        print(sorted(list(set(p._all_tile_positions))))
        self.assertEqual(p._all_tile_positions, [(0, (0, 0, 0)), (0, (1, 0, 0))])    


    def test_getitem(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        #batch size is 2, so the first 2 tiles go with the first batch
        #(size two), the third tile in in the second batch. the second
        #batch has only size 1 (is smaller than the specified batch size), 
        #because it contains the rest. 

        print(p[0].pixels().shape)
        print(p[1].pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 2)
        self.assertEqual(p[0].pixels().shape, (2, 3, 1, 6, 4))
        self.assertEqual(p[1].pixels().shape, (1, 3, 1, 6, 4))
       


    def test_getitem(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 3

        p = PredictionBatch(d, batch_size, size)
        #batch size is 3, this means all3 tempplates fit in one batch

        print(p[0].pixels().shape)
        print(len(p))
        self.assertEqual(len(p), 1)
        self.assertEqual(p[0].pixels().shape, (3, 3, 1, 6, 4))
        

    def test_get_actual_batch_size(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)


        self.assertEqual(p[0].get_actual_batch_size(), 2)
        self.assertEqual(p[1].get_actual_batch_size(), 1)

    def test_get_curr_tile_indices(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        c = TiffConnector(img_path, label_path)

        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2
        p = PredictionBatch(d, batch_size, size)


        self.assertEqual(p[0]._get_curr_tile_indices(), [0, 1])
        self.assertEqual(p[1]._get_curr_tile_indices(), [2])

    def test_put_probmap_data(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = tempfile.TemporaryDirectory()
        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        
        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((1, 2, 1, 6, 4))
        p[0].put_probmap_data(data)
        p[1].put_probmap_data(data)
        p[2].put_probmap_data(data)

    def test_put_probmap_data_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '/path/to/nowhere')
        savepath = tempfile.TemporaryDirectory()
        c = TiffConnector(img_path, label_path, savepath=savepath.name)

        d = Dataset(c)

        size = (1, 6, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2, 2, 1, 6, 4))
        p[0].put_probmap_data(data)

        data = np.ones((1, 2, 1, 6, 4))
        p[1].put_probmap_data(data)


    def test_put_probmap_data_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*')
        savepath = tempfile.TemporaryDirectory()
        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        size = (1, 3, 4)
        batch_size = 2

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((2, 3, 1, 3, 4))
        p[0].put_probmap_data(data)

        data = np.ones((2, 3, 1, 3, 4))
        p[1].put_probmap_data(data)  

        data = np.ones((2, 3, 1, 3, 4))
        p[2].put_probmap_data(data)       
    


    def test_put_probmap_data_multichannel_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*')
        savepath = tempfile.TemporaryDirectory()
        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        print('labels')
        print(c.labelvalue_mapping)
        original_labels = c.original_label_values_for_all_images()
        res = c.calc_label_values_mapping(original_labels)
        print('labels')
        print(c.labelvalue_mapping)
        d = Dataset(c)
        
        size = (1, 3, 4)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)
        print('labels')
        print(p.labels)

        data = np.ones((1, 6, 1, 3, 4))
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

            tile_size = (1, 5, 4) # size of network output layer in zxy
            padding = (0, 0, 0) # padding of network input layer in zxy, in respect to output layer

             # make training_batch mb and prediction interface p with TiffConnector binding
            c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath)
            p = PredictionBatch(Dataset(c), 2, tile_size, padding_zxy=padding)

            self.assertEqual(len(p), 255)
            self.assertEqual(p.labels, {1, 2, 3})

            #classify the whole bound dataset
            for counter, item in enumerate(p):
                pixels = item.pixels() #input for classifier
                mock_classifier_result = classify(pixels, counter) #classifier output
                #pass classifier results for each class to data source
                item.put_probmap_data(mock_classifier_result)     


    def test_put_probmap_data_for_label(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*')
        savepath = tempfile.TemporaryDirectory()
        c = TiffConnector(img_path, label_path, savepath=savepath.name)
        d = Dataset(c)

        d = Dataset(c)

        size = (3, 3, 3)
        batch_size = 1

        p = PredictionBatch(d, batch_size, size)

        data = np.ones((3, 3, 3))
        
        path1 = savepath.name + '40width26height3slices_rgb_class_109.tif'
        path2 = savepath.name + '40width26height6slices_rgb_class_109.tif'
        path3 = savepath.name + '6width4height3slices_rgb_class_109.tif'
        
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
        print(p._all_tile_positions)
        p._put_probmap_data_for_label(data, label=1, tile_pos_index=0) #first
        p._put_probmap_data_for_label(data, label=1, tile_pos_index=381) #last

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

