from unittest import TestCase
import os
import tempfile

import numpy as np
import yapic_io.image_importers as ip
import logging
logger = logging.getLogger(os.path.basename(__file__))
from tifffile import imsave, imread
base_path = os.path.dirname(__file__)

class TestImageImports(TestCase):
    def test_get_tiff_image_dimensions(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 3, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 2, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (3, 1, 6, 4))

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path, zstack=True)
        self.assertEqual(dims, (1, 2, 6, 4))
        

    # def test_get_tiff_image_dimensions_2(self):
    #     path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/40width26height3slices_rgb.tif')
    #     #dims = ip.import_tiff_image_2(path).shape
    #     dims = ip.get_tiff_image_dimensions(path, nr_channels=2)
    #     self.assertEqual(dims, (2, 3, 40, 26))

    # def test_get_tiff_image_dimensions_3(self):
    #     path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/40width26height3slices_rgb.tif')
    #     dims = ip.import_tiff_image_2(path).shape
    #     self.assertEqual(dims, (1, 3, 40, 26))


    def test_is_rgb_image(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        self.assertTrue(ip.is_rgb_image(path))

        path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/40width26height3slices_rgb.tif')
        self.assertFalse(ip.is_rgb_image(path))

    def test_imread_1(self):
        path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/40width26height3slices_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('2channel mask, three slices')
        print(im.shape)
        self.assertEqual(dims, (2, 3, 40, 26))  

    def test_imread_2(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('rgb, one z slice')
        print(im.shape)
        self.assertEqual(dims, (3, 1, 6, 4))

    def test_imread_3(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        im = ip.import_tiff_image(path, zstack=True)
        dims = im.shape
        print('grayscale, three z slices')
        print(im.shape)
        self.assertEqual(dims, (1, 3, 6, 4))

    def test_imread_3(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_1slice_2channels.tif')
        im = ip.import_tiff_image(path, multichannel=True)
        dims = im.shape
        print('grayscale, 1 slice, 2 channels')
        print(im.shape)
        self.assertEqual(dims, (2, 1, 6, 4))

    def test_imread_4(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_1slice_1channel.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('grayscale, 1 slice, 1 channel')
        print(im.shape)
        self.assertEqual(dims, (1, 1, 6, 4))

    def test_imread_5(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape
        print('rgb, 3 slice, 3 channel')
        print(im.shape)
        self.assertEqual(dims, (3, 3, 6, 4))                            

    def test_get_tiff_image_dimensions_4(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        im = ip.import_tiff_image(path)
        dims = im.shape


        
        self.assertEqual(dims, (3, 1, 6, 4))               



    def test_import_tiff_image(self):
        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_3slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_rgb_zstack.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_rgb.tif')
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)

        path = os.path.join(base_path, '../test_data/tif_images/6width_4height_2slices_8bit_grayscale_zstack.tif')
        
        im = ip.import_tiff_image(path, zstack=True)
        self.assertEqual((1, 2, 6, 4), im.shape)
        
        



    def test_import_tiff_image_2(self):
        path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/6width4height3slices_rgb.tif')
        im = ip.import_tiff_image(path)
        print(im.shape)
        print(np.unique(im.flatten()))
        self.assertEqual((2, 3, 6, 4), im.shape)

    def test_init_empty_tiff_image(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, 'empty.tif')
        
        
        
        xsize = 70
        ysize = 50
        ip.init_empty_tiff_image(path, xsize, ysize)

        img2 = ip.import_tiff_image(path)

        print(img2)
        print(img2.shape)
        self.assertEqual(img2.shape, (1, 1, 70, 50))
        print(np.unique(img2[:]))
        print (type(np.unique(img2[:])[0]))
        self.assertTrue(isinstance(np.unique(img2[:])[0], np.float32))


        try:
            os.remove(path)
        except:
            pass


    
    def test_init_empty_tiff_image_3d(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, 'empty_2.tif')

        xsize = 70
        ysize = 50
        zsize = 5
        ip.init_empty_tiff_image(path, xsize, ysize, z_size=zsize)

        img2 = ip.import_tiff_image(path, zstack=True)

        self.assertEqual(img2.shape, (1, 5, 70, 50))
        print(np.unique(img2[:]))
        print (type(np.unique(img2[:])[0]))
        self.assertTrue(isinstance(np.unique(img2[:])[0], np.float32))

        try:
            os.remove(path)
        except:
            pass


    def test_autocomplete(self):
        path =  'path/to/image'

        self.assertEqual('path/to/image.tif', ip.autocomplete_filename_extension(path, '.tif'))


    def test_add_vals_to_tiff_image(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, 'tile.tif')

        try:
            os.remove(path)
        except:
            pass  

        xsize = 5
        ysize = 7
        
        ip.init_empty_tiff_image(path, xsize, ysize)

        pos = (0, 1, 2)

        pixels = np.array([[[0.1, 0.2, 0.3, 0.4], \
                           [0.5, 0.6, 0.7, 0.8]]])

        
        val = np.array(\
         [[[[0, 0, 0, 0, 0, 0, 0], \
            [0, 0, .1, .2, .3, .4, 0], \
            [0, 0, .5, .6, .7, .8, 0], \
            [0, 0, 0, 0, 0, 0, 0], \
            [0, 0, 0, 0, 0, 0, 0]]]]\
            , dtype=np.float32)

        ip.add_vals_to_tiff_image(path, pos, pixels)
        img = ip.import_tiff_image(path)
        
        print(val)
        print(img)
        print(val.shape)
        print(img.shape)
        #r_img = np.around(img, decimals=2)
        print(type(val[0, 0, 0, 0]))
        print(type(img[0, 0, 0, 0]))
        print(type(val))
        print(type(img))
        print(np.testing.assert_equal(img, val))

        #img = ip.import_tiff_image(path)
        #print img 

        try:
            os.remove(path)
        except:
            pass


    def test_add_vals_to_tiff_image_3d(self):
        tmp = tempfile.TemporaryDirectory()
        path = os.path.join(tmp.name, 'tile.tif')

        try:
            os.remove(path)
        except:
            pass
        
        xsize = 5
        ysize = 7
        zsize = 2
        
        ip.init_empty_tiff_image(path, xsize, ysize, z_size=zsize)

        pos = (0, 1, 2)

        pixels = np.array([[[0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8]],
                           [[1.1, 1.2, 1.3, 1.4],
                            [1.5, 1.6, 1.7, 1.8]]
                           ])

        
        val = np.array(
             [[[[0, 0, 0, 0, 0, 0, 0],
                [0, 0, .1, .2, .3, .4, 0],
                [0, 0, .5, .6, .7, .8, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]],
               [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1.1, 1.2, 1.3, 1.4, 0],
                [0, 0, 1.5, 1.6, 1.7, 1.8, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]]
         ]], dtype=np.float32)

        ip.add_vals_to_tiff_image(path, pos, pixels)
        img = ip.import_tiff_image(path, zstack=True)

        print(val)
        print(img)
        #r_img = np.around(img, decimals=2)
        print(type(val[0, 0, 0, 0]))
        print(type(img[0, 0, 0, 0]))
        print(type(val))
        print(type(img))
        print(val==img)

        print(pixels.shape)

        np.testing.assert_equal(img, val)

        #img = ip.import_tiff_image(path)
        #print img 

        try:
            os.remove(path)
        except:
            pass        

