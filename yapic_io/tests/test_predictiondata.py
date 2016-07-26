from unittest import TestCase
import os
from yapic_io.tiffconnector import Tiffconnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_data import Prediction_data

import yapic_io.utils as ut
base_path = os.path.dirname(__file__)

class TestPredictiondata(TestCase):
    def test_computepos(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        
        c.filenames = [\
                ['6width4height3slices_rgb.tif', '6width4height3slices_rgb.tif']\
                ]
        c.load_label_filenames()    

        d = Dataset(c)

        size = (1,1,1)

        p = Prediction_data(d, size)

        print(p.tpl_pos)
        #self.assertTrue(False)