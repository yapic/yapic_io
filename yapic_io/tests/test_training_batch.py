from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset

import yapic_io.training_batch as mb 
from yapic_io.training_batch import TrainingBatch
base_path = os.path.dirname(__file__)

class TestTrainingBatch(TestCase):
    def test_pick_random_tpl(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path,label_path)
        

        d = Dataset(c)

        size = (1,3,4)
        pad = (1,2,2)
        
        batch_size = 4

        m = TrainingBatch(d, batch_size, size, padding_zxy=pad)

        tpl = m._pick_random_tpl()
        print(tpl)

        print(tpl.pixels.shape)
        print(tpl.weights.shape)

        #self.assertTrue(False)


    def test_getitem(self):

        from yapic_io.factories import make_tiff_interface
        #define data loacations
        pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
        savepath = 'yapic_io/test_data/tmp/'
        
        tpl_size = (1,5,4) # size of network output layer in zxy
        padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
        # make training_batch mb and prediction interface p with TiffConnector binding
        m, p = make_tiff_interface(pixel_image_dir, label_image_dir\
            , savepath, tpl_size, padding_zxy=padding, training_batch_size=6) 
        counter=0
        for mini in m:
            weights = mini.weights #shape is (6,3,1,5,4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
            pixels = mini.pixels # shape is (6,3,1,9,8) : batchsize 6, 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
            self.assertEqual(weights.shape, (6,3,1,5,4))
            self.assertEqual(pixels.shape, (6,3,1,9,8))
            #here: apply training on mini.pixels and mini.weights
            counter += 1
            if counter > 10: #m is infinite
                break


    def test_getitem_multichannel_labels(self):

        from yapic_io.factories import make_tiff_interface
        #define data loacations
        pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
        label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels_multichannel/*.tif'
        savepath = 'yapic_io/test_data/tmp/'
        
        tpl_size = (1,5,4) # size of network output layer in zxy
        padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
        # make training_batch mb and prediction interface p with TiffConnector binding
        m, p = make_tiff_interface(pixel_image_dir, label_image_dir\
            , savepath, tpl_size, padding_zxy=padding, training_batch_size=6) 
        counter=0
        for mini in m:
            weights = mini.weights #shape is (6,6,1,5,4) : batchsize 6 , 6 label-classes, 1 z, 5 x, 4 y
            pixels = mini.pixels # shape is (6,3,1,9,8) : batchsize 6, 6 channels, 1 z, 9 x, 4 y (more xy due to padding)
            self.assertEqual(weights.shape, (6,6,1,5,4))
            self.assertEqual(pixels.shape, (6,3,1,9,8))
            #here: apply training on mini.pixels and mini.weights
            counter += 1
            if counter > 10: #m is infinite
                break        


        
        #self.assertTrue(False)    

