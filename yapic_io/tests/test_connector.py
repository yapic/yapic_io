from unittest import TestCase
import os

from numpy.testing import assert_array_equal
from yapic_io.tiff_connector import TiffConnector
from yapic_io.coordinate_connector import Coordinate_connector
import logging

logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)

class TestConnector(TestCase):
    
    def test_tiff_connector_training_tile_mode(self):
        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/*.tif')
        c = TiffConnector(img_path, 'path/to/nowhere/')

        self.assertTrue(c.tile_fetching_mode,'polling')

    def test_coordinate_connector(self):

        
        class TestConnClass(Coordinate_connector):
            def __init__(self):
                super().__init__() #sets training_tile_mode

            def image_count(self):
                return 1    

            def label_count_for_image(self, image_nr):
                return image_nr

            def get_tile(self, image_nr=None, pos=None, size=None):
                return image_nr

            def label_tile(self, image_nr, pos_zxy, 
                           size_zxy, label_value):
                return image_nr
            
            def put_tile(self, pixels, pos_zxy,
                         image_nr, label_value):
                return pixels
            
            def image_dimensions(self, image_nr):
                return image_nr
            
            def label_index_to_coordinate(self, image_nr,
                                          label_value, label_index):
                return image_nr


        c = TestConnClass()

        self.assertTrue(c.tile_fetching_mode,'by_label_index')

    