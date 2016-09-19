import os
from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset

import yapic_io.training_batch as mb 
from yapic_io.training_batch import TrainingBatch
base_path = os.path.dirname(__file__)


from yapic_io.factories import make_tiff_interface
#define data loacations
pixel_image_dir = 'yapic_io/test_data/milt/cell_pixels.tif'
label_image_dir = 'yapic_io/test_data/milt/cell_labels.tif'
savepath = 'yapic_io/test_data/tmp/'



tpl_size = (1,50,50) # size of network output layer in zxy
padding = (0,10,10) # padding of network input layer in zxy, in respect to output layer
# make training_batch mb and prediction interface p with TiffConnector binding
m, p = make_tiff_interface(pixel_image_dir, label_image_dir\
    , savepath, tpl_size, padding_zxy=padding, training_batch_size=3, zstack=False, multichannel_label_image=True) 

mini = next(m)

mini.weights()
mini.pixels()

