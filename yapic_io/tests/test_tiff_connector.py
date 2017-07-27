from unittest import TestCase
import os
from pprint import pprint
import numpy as np
from numpy.testing import assert_array_equal
from yapic_io.tiff_connector import TiffConnector
import yapic_io.image_importers as ip
import logging
from pprint import pprint
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestTiffconnector(TestCase):
    def test_load_filenames(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        c = TiffConnector(img_path, 'path/to/nowhere/')

        img_filenames = ['6width4height3slices_rgb.tif', 
                         '40width26height3slices_rgb.tif', 
                         '40width26height6slices_rgb.tif']

        fnames = [e[0] for e in c.filenames]
        self.assertEqual(set(img_filenames), set(fnames))


    def test_load_filenames_from_same_path(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/img*.tif')
        lbl_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/lbl*.tif')
        c = TiffConnector(img_path, lbl_path)
        
        expected_names = \
            [['img_40width26height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif']\
           , ['img_40width26height6slices_rgb.tif', None]\
           , ['img_6width4height3slices_rgb.tif', 'lbl_6width4height3slices_rgb.tif']]

        self.assertEqual(c.filenames, expected_names)   


    def test_filter_labeled(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/img*.tif')
        lbl_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/lbl*.tif')
        c = TiffConnector(img_path, lbl_path).filter_labeled()

        expected_names = \
            [('img_40width26height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif')\
           , ('img_6width4height3slices_rgb.tif', 'lbl_6width4height3slices_rgb.tif')]

        self.assertEqual(set(c.filenames), set(expected_names))


    def test_split(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/img*.tif')
        lbl_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/lbl*.tif')
        c = TiffConnector(img_path, lbl_path)
        c1, c2 = c.split(0.5)

        expected_names1 = [('img_40width26height3slices_rgb.tif', 'lbl_40width26height3slices_rgb.tif')]
        expected_names2 = [('img_40width26height6slices_rgb.tif', None), 
                           ('img_6width4height3slices_rgb.tif', 'lbl_6width4height3slices_rgb.tif')]

        self.assertEqual(set(c1.filenames), set(expected_names1))
        self.assertEqual(set(c2.filenames), set(expected_names2))

        #test for issue #1
        self.assertEqual(c1.labelvalue_mapping, c.labelvalue_mapping)
        self.assertEqual(c2.labelvalue_mapping, c.labelvalue_mapping)


    def test_load_filenames_emptyfolder(self):
        img_path = os.path.join(base_path, '../test_data/empty_folder/')
        c = TiffConnector(img_path, 'path/to/nowhere/')
        self.assertEqual(len(c.filenames), 0)


    def test_image_dimensions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/img*.tif')
        lbl_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/lbl*.tif')
        c = TiffConnector(img_path, lbl_path)
        
        with self.assertRaises(IndexError):
            c.image_dimensions(4)
        
        self.assertEqual(c.image_dimensions(0), (3, 3, 40, 26))
        self.assertEqual(c.image_dimensions(1), (3, 6, 40, 26))
        self.assertEqual(c.image_dimensions(2), (3, 3, 6, 4))   


    def test_load_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/*.tif')
        c = TiffConnector(img_path, 'path/to/nowhere/')
        
        im = c.load_image(0)
        self.assertEqual(im.shape, (3, 3, 40, 26))


    def test_get_tile(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/together/*.tif')
        c = TiffConnector(img_path, 'path/to/nowhere/')
        
        image_nr = 0
        pos = (0, 0, 0, 0)
        size = (1, 1, 1, 2)
        im = c.load_image(0)
        tile = c.get_tile(image_nr=image_nr, pos=pos, size=size)
        val = np.empty(shape=size)
        val[0, 0, 0, 0] = 151
        val[0, 0, 0, 1] = 151
        val = val.astype(int)
        print(val)
        print(tile)
        np.testing.assert_array_equal(tile, val)


    def test_exists_label_for_image_nr(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)
        
        self.assertTrue(c.exists_label_for_image(0))
        self.assertFalse(c.exists_label_for_image(1))
        self.assertTrue(c.exists_label_for_image(2))


    def test_load_label_filenames(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)
        
        self.assertEqual(c.filenames[0][1], '40width26height3slices_rgb.tif')
        self.assertIsNone(c.filenames[1][1])
        self.assertEqual(c.filenames[2][1], '6width4height3slices_rgb.tif')


    def test_load_label_matrix(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)

        im = c.load_image(2)
        labelmat = c.load_label_matrix(2)
        print(labelmat)
        self.assertEqual(labelmat.shape, (1, 3, 6, 4))


    def test_load_label_matrix_multichannel(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*.tif')

        c = TiffConnector(img_path, label_path)

        im = c.load_image(2)
        labelmat = c.load_label_matrix(2)
        self.assertEqual(labelmat.shape, (2, 3, 6, 4)) 


    def test_label_tile(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*.tif')

        c = TiffConnector(img_path, label_path)

        label_value = 2
        pos_zxy = (0, 0, 0)
        size_zxy = (1, 6, 4)

        tile = c.label_tile(2, pos_zxy, size_zxy, label_value)

        val_z0 = np.array(\
                [[[False, False, False, False], 
                  [False, False, False, False], 
                  [False, True, True, True], 
                  [False, True, True, True], 
                  [False, False, False, False], 
                  [False, False, False, False]]])
        assert_array_equal(val_z0, tile)

        pos_zxy = (1, 0, 0)
        size_zxy = (1, 6, 4)

        tile_z1 = c.label_tile(2, pos_zxy, size_zxy, label_value)

        val_z1 = np.array(\
        [[[False, False, False, False], 
          [False, False, False, False], 
          [False, False, False, False], 
          [False, False, False, False], 
          [False, False, False, False], 
          [ True, True, False, False]]])
        assert_array_equal(val_z1, tile_z1)


    def test_check_label_matrix_dimensions(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*.tif')

        c = TiffConnector(img_path, label_path)
        c.check_label_matrix_dimensions()


    def test_check_label_matrix_dimensions_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel_not_valid/')
    
        self.assertRaises(ValueError, lambda: TiffConnector(img_path, label_path))    


    def test_label_values_for_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)

        labelvals = c.label_values_for_image(2)
        self.assertEqual(labelvals, [2, 3])


    def test_label_count_for_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)

        count = c.label_count_for_image(2)
        print(c.labelvalue_mapping)

        self.assertEqual(count, {2 : 11, 3: 3})


    def test_get_probmap_path(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path, savepath = 'path/to/probmaps')
        
        p = c.get_probmap_path(2, 3)

        self.assertEqual(p, 'path/to/probmaps/6width4height3slices_rgb_class_3.tif')


    def test_init_probmap_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')
        savepath = os.path.join(base_path, '../test_data/tmp/')

        c = TiffConnector(img_path, label_path, savepath = savepath)

        fnames = ['6width4height3slices_rgb_class_1.tif', '6width4height3slices_rgb_class_2.tif']
        path1, path2 = [os.path.join(savepath, f) for f in fnames]

        for p in [path1, path2]:
            try:
                os.remove(path1)
            except: pass

        c.init_probmap_image(2, 1, overwrite=True)
        c.init_probmap_image(2, 2, overwrite=True)    
            
        probim_1 = ip.import_tiff_image(path1, zstack=True)
        probim_2 = ip.import_tiff_image(path2, zstack=True)

        self.assertEqual(probim_1.shape, (1, 3, 6, 4))
        self.assertEqual(probim_2.shape, (1, 3, 6, 4))

        for p in [path1, path2]:
            try:
                os.remove(path1)
            except: pass


    def test_put_tile_1(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')
        savepath = os.path.join(base_path, '../test_data/tmp/')

        c = TiffConnector(img_path, label_path, savepath=savepath)

        pixels = np.array([[[.1, .2, .3], \
                            [.4, .5, .6]]], dtype=np.float32)
        
        path = os.path.join(savepath, '6width4height3slices_rgb_class_3.tif')
        
        try:
            os.remove(path)
        except: pass    

        c.put_tile(pixels, pos_zxy=(0, 1, 1), image_nr=2, label_value=3)
        probim = ip.import_tiff_image(path, zstack=True)
        pprint(probim)

        val = \
            np.array([[[[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.1       , 0.2       , 0.3       ], \
             [ 0.        , 0.4       , 0.5       , 0.6       ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]]]], dtype=np.float32)
        
        self.assertTrue((val==probim).all())

        try:
            os.remove(path)
        except: pass    

        

    def test_put_tile_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')
        savepath = os.path.join(base_path, '../test_data/tmp/')

        c = TiffConnector(img_path, label_path, savepath=savepath)

        pixels = np.array([[[.1, .2, .3], \
                            [.4, .5, .6]]], dtype=np.float32)
        
        path = os.path.join(savepath, '6width4height3slices_rgb_class_3.tif')
        
        try:
            os.remove(path)
        except: pass    

        c.put_tile(pixels, pos_zxy=(0, 1, 1), image_nr=2, label_value=3)
        probim = ip.import_tiff_image(path, zstack=True)
        pprint(probim)

        val = \
            np.array([[[[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.1       , 0.2       , 0.3       ], \
             [ 0.        , 0.4       , 0.5       , 0.6       ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]]]], dtype=np.float32)
        
        self.assertTrue((val==probim).all())


        c.put_tile(pixels, pos_zxy=(2, 1, 1), image_nr=2, label_value=3)
        probim_2 = ip.import_tiff_image(path, zstack=True)
        pprint(probim_2)

        val_2 = \
            np.array([[[[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.1       , 0.2       , 0.3       ], \
             [ 0.        , 0.4       , 0.5       , 0.6       ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]], \
            [[ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.1       , 0.2       , 0.3        ], \
             [ 0.        , 0.4       , 0.5       , 0.6        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ], \
             [ 0.        , 0.        , 0.        , 0.        ]]]], dtype=np.float32)

        self.assertTrue((val_2==probim_2).all())    

        try:
            os.remove(path)
        except: pass    


    def test_original_label_values_for_image(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)

        res = c.original_label_values_for_image(2)
        self.assertEqual(res, [{109, 150}])

        res = c.original_label_values_for_image(1)
        self.assertIsNone(res)

        res = c.original_label_values_for_image(0)
        self.assertEqual(res, [{91, 109, 150}])


    def test_original_label_values_for_image_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*.tif')

        c = TiffConnector(img_path, label_path)

        res = c.original_label_values_for_image(0)
        self.assertEqual(res, [{91, 109, 150}, {91, 109, 150}])

        res = c.original_label_values_for_image(1)
        self.assertIsNone(res)

        res = c.original_label_values_for_image(2)
        self.assertEqual(res, [{109, 150}, {109, 150}])


    def test_original_label_values(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/*.tif')

        c = TiffConnector(img_path, label_path)

        res = c.original_label_values_for_all_images()
        self.assertEqual(res, [{91, 109, 150}, {91, 109, 150}])
        

    def test_map_label_values(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels_multichannel/')
        c = TiffConnector(img_path, label_path)

        res = c.map_label_values()
        self.assertEqual(res, [{91:1, 109:2, 150:3}, {91:4, 109:5, 150:6}])


    def test_map_label_values_2(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)

        res = c.map_label_values()
        self.assertEqual(res, [{91:1, 109:2, 150:3}])


    def test_map_label_values_3(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = TiffConnector(img_path, label_path)

        res = c.map_label_values()
        self.assertEqual(res, [{91:1, 109:2, 150:3}])


    def test_map_label_values_4(self):    
        
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/6width4height3slices_rgb.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')

        c = TiffConnector(img_path, label_path)

        c.map_label_values()
        self.assertEqual(c.labelvalue_mapping, [{109: 1, 150: 2}])

