from unittest import TestCase
import os
from yapic_io.tiffconnector import Tiffconnector
from yapic_io.dataset import Dataset

import yapic_io.minibatch as mb 
from yapic_io.minibatch import Minibatch
base_path = os.path.dirname(__file__)

class TestMinibatch(TestCase):
    def test_pick_random_tpl(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        

        d = Dataset(c)

        size = (1,3,4)
        pad = (1,2,2)
        
        batch_size = 4

        m = Minibatch(d, batch_size, size, padding_zxy=pad)

        tpl = m._pick_random_tpl()
        print(tpl)

        print(tpl.pixels.shape)
        print(tpl.weights.shape)

        #self.assertTrue(False)


    def test_getitem(self):

        img_path = os.path.join(base_path, '../test_data/tiffconnector_1/im/')
        label_path = os.path.join(base_path, '../test_data/tiffconnector_1/labels/')
        c = Tiffconnector(img_path,label_path)
        

        d = Dataset(c)

        size = (1,3,4)
        pad = (1,2,2)
        
        batch_size = 4

        m = Minibatch(d, batch_size, size, padding_zxy=pad)
        #m._fetch_minibatch_data()
        c=0
        for mini in m:
            print(mini.augmentations)

            c+=1
            if c>10:
                break


        
        #self.assertTrue(False)    

