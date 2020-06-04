from yapic_io.tiff_connector import TiffConnector
from yapic_io.ilastik_connector import IlastikConnector
from skimage import io
from functools import lru_cache
import numpy as np
from glob import glob
import os


class CellvoyConnector(IlastikConnector):

    def __init__(self, img_filepath, label_filepath, savepath=None):

        ref_names = glob(os.path.join(img_filepath, '*C01.tif'))
        self.names_all_channels = []
        for ref_name in ref_names:
            ref_name = ref_name[:-16]
            self.names_all_channels.append(sorted(glob(ref_name + '*C0*.tif')))

        super().__init__(ref_names,
                         label_filepath,
                         savepath=savepath)

        # order names_all_channels according to self.filenames
        pxnames_tiff_connector = [str(e.img) for e in self.filenames]
        pxnames_cellvoy = [os.path.basename(e[0])
                           for e in self.names_all_channels]
        idx = [pxnames_cellvoy.index(e) for e in pxnames_tiff_connector]
        self.names_all_channels = [self.names_all_channels[i] for i in idx]

    @lru_cache(maxsize=10)
    def _open_image_file(self, image_nr):

        img_names = self.names_all_channels[image_nr]
        pixels = np.array([[io.imread(img_name)] for img_name in img_names])
        pixels = np.expand_dims(pixels, axis=0)
        return pixels

    def image_dimensions(self, image_nr):
        dims = np.array(self. _open_image_file(image_nr)[0].shape)
        dims[[-1, -2]] = dims[[-2, -1]]
        return dims
