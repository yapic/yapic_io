from yapic_io.tiff_connector import TiffConnector
from skimage import io
from functools import lru_cache
import numpy as np
from glob import glob
import os


class CellvoyConnector(TiffConnector):

    def __init__(self, img_filepath, label_filepath, savepath=None):

        ref_names = glob(os.path.join(img_filepath, '*C01.tif'))
        self.names_all_channels = []
        for ref_name in ref_names:
            ref_name = ref_name[:-16]
            self.names_all_channels.append(sorted(glob(ref_name + '*C0*.tif')))

        super().__init__(ref_names,
                         label_filepath,
                         savepath=savepath)

    @lru_cache(maxsize=10)
    def _open_image_file(self, image_nr):

        path = self.img_path / self.filenames[image_nr].img

        img_names = self.names_all_channels[image_nr]

        pixels = np.array([[io.imread(img_name)] for img_name in img_names])
        pixels = np.swapaxes(pixels, -1, -2)
        print(pixels.shape)
        pixels = np.expand_dims(pixels, axis=0)
        print(pixels.shape)
        return pixels
