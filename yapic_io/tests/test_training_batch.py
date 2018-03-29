from unittest import TestCase
import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.ilastik_connector import IlastikConnector
from yapic_io.dataset import Dataset

from yapic_io.training_batch import TrainingBatch
from pprint import pprint
import numpy as np
import tempfile
from numpy.testing import assert_array_equal, assert_array_almost_equal

base_path = os.path.dirname(__file__)

class TestTrainingBatch(TestCase):

    def test_get_ilastik_weights(self):
        img_path = os.path.join(base_path, '../test_data/ilastik')
        lbl_path = os.path.join(base_path, '../test_data/ilastik/ilastik-1.2.ilp')

        c = IlastikConnector(img_path, lbl_path)

        d = Dataset(c)

        size = (8, 2, 4)
        pad = (0, 0, 0)

        batch_size = 1

        m = TrainingBatch(d, size, padding_zxy=pad)
        m.augment_by_flipping(False)
        mini = next(m)
        weights = mini.weights()

        #label 1 at position (4,1,3)
        self.assertTrue(weights[0, 0, 4, 1, 3] == 1)

        #label 2 at position (1,1,1)
        self.assertTrue(weights[0, 1, 1, 1, 1] == 1)

        #label 3 at position (1,1,1)
        self.assertTrue(weights[0, 2, 7, 1, 1] == 1)





    def test_random_tile(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)


        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)

        batch_size = 4

        m = TrainingBatch(d, size, padding_zxy=pad)

        tile = m._random_tile()


    def test_getitem(self):
        from yapic_io import TiffConnector, Dataset, PredictionBatch
        #define data loacations
        pixel_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_image_dir = os.path.join(base_path, '..//test_data/tiffconnector_1/labels/')
        savepath = tempfile.TemporaryDirectory()

        tile_size = (1, 5, 4) # size of network output layer in zxy
        padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer
        c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath.name)
        m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)

        for counter, mini in enumerate(m):
            weights = mini.weights() #shape is (6, 3, 1, 5, 4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
            pixels = mini.pixels() # shape is (6, 3, 1, 9, 8) : batchsize 6, 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
            self.assertEqual(weights.shape, (3, 3, 1, 5, 4))
            self.assertEqual(pixels.shape, (3, 3, 1, 9, 8))
            #here: apply training on mini.pixels and mini.weights
            if counter > 10: #m is infinite
                break


    def test_getitem_multichannel_labels(self):
        from yapic_io import TiffConnector, Dataset, PredictionBatch
        #define data loacations
        pixel_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/')
        savepath = tempfile.TemporaryDirectory()

        tile_size = (1, 5, 4) # size of network output layer in zxy
        padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer
        # make training_batch mb and prediction interface p with TiffConnector binding
        c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath.name)
        m = TrainingBatch(Dataset(c), tile_size, padding_zxy=padding)

        for counter, mini in enumerate(m):
            weights = mini.weights() #shape is (6, 6, 1, 5, 4) : batchsize 6 , 6 label-classes, 1 z, 5 x, 4 y
            pixels = mini.pixels() # shape is (6, 3, 1, 9, 8) : batchsize 6, 6 channels, 1 z, 9 x, 4 y (more xy due to padding)
            self.assertEqual(weights.shape, (6, 6, 1, 5, 4))
            self.assertEqual(pixels.shape, (6, 3, 1, 9, 8))
            #here: apply training on mini.pixels and mini.weights
            if counter > 10: #m is infinite
                break


    def test_init_trainingbatch(self):
        pixel_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_image_dir = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/')
        savepath = tempfile.TemporaryDirectory()

        tile_size = (1, 5, 4) # size of network output layer in zxy
        padding = (0, 2, 2) # padding of network input layer in zxy, in respect to output layer

        c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath.name)

        d = Dataset(c)


    def test_normalize_zscore(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)


        d = Dataset(c)

        size = (1, 2, 2)
        pad = (0, 0, 0)

        batchsize = 2
        nr_channels = 3
        nz = 1
        nx = 4
        ny = 5

        m = TrainingBatch(d, size, padding_zxy=pad)

        m.set_normalize_mode('local_z_score')

        pixels = np.zeros((batchsize, nr_channels,nz,nx,ny))
        pixels[:,0,:,:,:] = 1
        pixels[:,1,:,:,:] = 2
        pixels[:,2,:,:,:] = 3
        #pprint(pixels)
        #print(pixels.shape)
        p_norm = m._normalize(pixels)
        self.assertTrue((p_norm == 0).all())

        #add variation
        pixels[:,0,0,0,0] = 2
        pixels[:,0,0,0,1] = 0

        pixels[:,1,0,0,0] = 3
        pixels[:,1,0,0,1] = 1

        pixels[:,2,0,0,0] = 4
        pixels[:,2,0,0,1] = 2

        p_norm = m._normalize(pixels)
        #pprint(p_norm)


        assert_array_equal(p_norm[:,0,:,:,:], p_norm[:,1,:,:,:])
        assert_array_equal(p_norm[:,0,:,:,:], p_norm[:,2,:,:,:])

        val = np.array([[[ 3.16227766, -3.16227766, 0., 0., 0.],
                         [ 0.        ,  0.        , 0., 0., 0.],
                         [ 0.        ,  0.        , 0., 0., 0.],
                         [ 0.        ,  0.        , 0., 0., 0.]]])

        assert_array_almost_equal(val, p_norm[0,0,:,:,:])


    def test_normalize_global(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)


        d = Dataset(c)

        size = (1, 2, 2)
        pad = (0, 0, 0)

        batchsize = 2
        nr_channels = 3
        nz = 1
        nx = 4
        ny = 5


        val = np.array(\
         [[[ 0.33333333,  0.33333333,  0.33333333,  0.33333333,  0.33333333],
           [ 0.33333333,  0.33333333,  0.33333333,  0.33333333,  0.33333333],
           [ 0.33333333,  0.33333333,  0.33333333,  0.33333333,  0.33333333],
           [ 0.33333333,  0.33333333,  0.33333333,  0.33333333,  0.33333333]],
          [[ 0.66666667,  0.66666667,  0.66666667,  0.66666667,  0.66666667],
           [ 0.66666667,  0.66666667,  0.66666667,  0.66666667,  0.66666667],
           [ 0.66666667,  0.66666667,  0.66666667,  0.66666667,  0.66666667],
           [ 0.66666667,  0.66666667,  0.66666667,  0.66666667,  0.66666667]],
          [[ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ],
           [ 1.        ,  1.        ,  1.        ,  1.        ,  1.        ]]])


        m = TrainingBatch(d, size, padding_zxy=pad)

        m.set_normalize_mode('global', minmax=[0,3])

        pixels = np.zeros((batchsize, nr_channels,nz,nx,ny))
        pixels[:,0,:,:,:] = 1
        pixels[:,1,:,:,:] = 2
        pixels[:,2,:,:,:] = 3
        #pprint(pixels)
        #print(pixels.shape)
        p_norm = m._normalize(pixels)

        pprint(p_norm)
        print(pixels.shape)
        print(p_norm.shape)

        assert_array_almost_equal(val, p_norm[0,:,0,:,:])



    def test_set_augmentation(self):


        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)


        d = Dataset(c)

        size = (1, 3, 4)
        pad = (1, 2, 2)

        batch_size = 4

        m = TrainingBatch(d, size, padding_zxy=pad)


        print(m.augmentation)
        self.assertEqual(m.augmentation, {'flip'})

        m.augment_by_rotation(True)
        print(m.augmentation)
        self.assertEqual(m.augmentation, {'flip', 'rotate'})
        self.assertEqual(m.rotation_range,(-45,45))

        m.augment_by_shear(True)
        print(m.augmentation)
        self.assertEqual(m.augmentation, {'flip', 'rotate', 'shear'})
        self.assertEqual(m.shear_range,(-5,5))

        m.augment_by_flipping(False)
        print(m.augmentation)
        self.assertEqual(m.augmentation, {'rotate', 'shear'})
