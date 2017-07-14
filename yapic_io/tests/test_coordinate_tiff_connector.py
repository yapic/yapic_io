from unittest import TestCase
import os
from pprint import pprint
import numpy as np
from numpy.testing import assert_array_equal
from yapic_io.coordinate_tiff_connector import CoordinateTiffConnector
import yapic_io.image_importers as ip
import logging
from pprint import pprint
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestCoordinateTiffconnector(TestCase):


    def test_label_index_to_coordinate(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/*.tif')
        
        c = CoordinateTiffConnector(img_path, label_path)

        label_value = 2 #wrong labelvalue
        label_index = 10
        image_nr = 2
        print(c.labelvalue_mapping)
        
        #self.assertTrue(c.label_index_to_coordinate(image_nr, label_value, 10)
        assert_array_equal(c.label_index_to_coordinate(image_nr, label_value, 10), np.array([0, 2, 1, 2]))
        assert_array_equal(c.label_index_to_coordinate(image_nr, label_value, 9), np.array([0, 2, 1, 0]))
        assert_array_equal(c.label_index_to_coordinate(image_nr, label_value, 0), np.array([0, 0, 2, 1]))    
        #self.assertTrue(False)    