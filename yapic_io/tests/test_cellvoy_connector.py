import itertools
from unittest import TestCase
import os
import numpy as np
from numpy.testing import assert_array_equal
from yapic_io.tiff_connector import TiffConnector
from yapic_io.cellvoy_connector import CellvoyConnector
import yapic_io.cellvoy_connector as cv
import logging
import tempfile
from pathlib import Path
logger = logging.getLogger(os.path.basename(__file__))

base_path = os.path.dirname(__file__)
data_dir = os.path.join(base_path, '../test_data/cellvoyager')


class TestCellvoyConnector(TestCase):

    def test_init_cellvoy(self):
        img_files = [ os.path.join(data_dir, 'GASPR01S01R01p01E01CD_A05_T0001F001L01A01Z01C01.tif'),
                      os.path.join(data_dir, 'GASPR01S01R02p01E01CD_A06_T0001F012L01A01Z01C01.tif')]
        cc = CellvoyConnector(data_dir, 'some/path')
        print('imgdims')
        print(cc.image_dimensions(0))
        print('hallpo')
        px = cc.get_tile(0, (0,0,0,0), (10, 10, 2000, 2000))

        print(px.shape)
        assert False
