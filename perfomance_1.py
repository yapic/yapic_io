import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset

import yapic_io.training_batch as mb 
from yapic_io.training_batch import TrainingBatch
base_path = os.path.dirname(__file__)

from yapic_io import TiffConnector, Dataset, PredictionBatch

#define data loacations
pixel_image_dir = 'yapic_io/test_data/milt/cell_pixels.tif'
label_image_dir = 'yapic_io/test_data/milt/cell_labels.tif'
savepath = 'yapic_io/test_data/tmp/'



tpl_size = (1,50,50) # size of network output layer in zxy
padding = (0,10,10) # padding of network input layer in zxy, in respect to output layer
# make training_batch mb and prediction interface p with TiffConnector binding

c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath)
m = TrainingBatch(Dataset(c), tpl_size, padding_zxy=padding)

mini = next(m)

mini.weights()
mini.pixels()

