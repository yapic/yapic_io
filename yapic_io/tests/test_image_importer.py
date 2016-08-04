from unittest import TestCase
import os

import numpy as np
import yapic_io.image_importers as ip
import logging
logger = logging.getLogger(os.path.basename(__file__))

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
        dims = ip.get_tiff_image_dimensions(path)
        self.assertEqual(dims, (1, 2, 6, 4))
        

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
        dims = ip.get_tiff_image_dimensions(path)
        im = ip.import_tiff_image(path)
        self.assertEqual(dims, im.shape)


    def test_init_empty_tiff_image(self):
        path = os.path.join(base_path, '../test_data/tmp/empty.tif')
        
        
        
        xsize = 70
        ysize = 50
        ip.init_empty_tiff_image(path, xsize, ysize)

        img2 = ip.import_tiff_image(path)

        print(img2)
        print(img2.shape)
        self.assertEqual(img2.shape, (1,1,70,50))
        print(np.unique(img2[:]))
        print (type(np.unique(img2[:])[0]))
        self.assertTrue(isinstance(np.unique(img2[:])[0], np.float32))


        try:
            os.remove(path)
        except:
            pass


    
    def test_init_empty_tiff_image_3d(self):
        path = os.path.join(base_path, '../test_data/tmp/empty_2.tif')

        xsize = 70
        ysize = 50
        zsize = 5
        ip.init_empty_tiff_image(path, xsize, ysize, z_size=zsize)

        img2 = ip.import_tiff_image(path)

        print(img2)
        print(img2.shape)
        self.assertEqual(img2.shape, (1,5,70,50))
        print(np.unique(img2[:]))
        print (type(np.unique(img2[:])[0]))
        self.assertTrue(isinstance(np.unique(img2[:])[0], np.float32))


        try:
            os.remove(path)
        except:
            pass


                


    def test_autocomplete(self):
        path =  'path/to/image'

        self.assertEqual('path/to/image.tif', ip.autocomplete_filename_extension(path))


    def test_add_vals_to_tiff_image(self):
        path = os.path.join(base_path, '../test_data/tmp/tpl.tif')

        try:
            os.remove(path)
        except:
            pass  

        xsize = 5
        ysize = 7
        
        ip.init_empty_tiff_image(path, xsize, ysize)

        pos = (0,1,2)

        pixels = np.array([[[0.1,0.2,0.3,0.4],\
                           [0.5,0.6,0.7,0.8]]])

        
        val = np.array(\
         [[[[0, 0, 0, 0, 0, 0, 0],\
            [0, 0, .1,.2,.3,.4,0],\
            [0, 0, .5,.6,.7,.8,0],\
            [0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0]]]]\
            , dtype=np.float32)

        ip.add_vals_to_tiff_image(path, pos, pixels)
        img = ip.import_tiff_image(path)
        
        print(val)
        print(img)
        print(val.shape)
        print(img.shape)
        #r_img = np.around(img, decimals=2)
        print(type(val[0,0,0,0]))
        print(type(img[0,0,0,0]))
        print(val==img)

        self.assertTrue((val==img).all())

        #img = ip.import_tiff_image(path)
        #print img 

        try:
            os.remove(path)
        except:
            pass


    def test_add_vals_to_tiff_image_3d(self):
        path = os.path.join(base_path, '../test_data/tmp/tpl.tif')

        try:
            os.remove(path)
        except:
            pass  
        

        xsize = 5
        ysize = 7
        zsize = 2
        
        ip.init_empty_tiff_image(path, xsize, ysize, z_size=zsize)

        pos = (0,1,2)

        pixels = np.array([[[0.1,0.2,0.3,0.4],\
                            [0.5,0.6,0.7,0.8]],\
                           [[1.1,1.2,1.3,1.4],\
                            [1.5,1.6,1.7,1.8]]\
                           ])

        
        val = np.array(\
         [[[[0, 0, 0, 0, 0, 0, 0],\
            [0, 0, .1,.2,.3,.4,0],\
            [0, 0, .5,.6,.7,.8,0],\
            [0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0]],\
           [[0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 1.1,1.2,1.3,1.4,0],\
            [0, 0, 1.5,1.6,1.7,1.8,0],\
            [0, 0, 0, 0, 0, 0, 0],\
            [0, 0, 0, 0, 0, 0, 0]]\
         ]], dtype=np.float32)

        ip.add_vals_to_tiff_image(path, pos, pixels)
        img = ip.import_tiff_image(path)
        
        print(val)
        print(img)
        print(val.shape)
        print(img.shape)
        #r_img = np.around(img, decimals=2)
        print(type(val[0,0,0,0]))
        print(type(img[0,0,0,0]))
        print(val==img)

        self.assertTrue((val==img).all())

        #img = ip.import_tiff_image(path)
        #print img 

        try:
            os.remove(path)
        except:
            pass        


