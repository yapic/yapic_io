from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_data import PredictionData
from yapic_io.minibatch import Minibatch


def make_tiff_interface(img_filepath, label_filepath, savepath,\
		tpl_size_zxy, minibatch_size=10, padding_zxy=(0,0,0)\
		, multichannel_pixel_image=None\
        , multichannel_label_image=None\
        , zstack=True):
	
	'''
	Convenience method to make a Minibatch and a PredictionData
	object bound to data from a TiffConnector

	:param img_filepath: path to pixelimages, filter can be applied with wildcards
	:param label_filepath: path to labelimages, filter can be applied with wildcards
	:param savepath: path for saving probability images (classification result)
	:param tpl_size_zxy: size of output layer
	:param minibatch_size: size of minibatch for training
	:param padding_zxy: growing of input layer in (z,x,y) compared to output layer
	:param multichannel_pixel_image: if True pixel images are interpreted as multichannel (rather than multi z slice)
	:type multichannel_pixel_image: bool
	:param multichannel_label_image: if True label images are interpreted as multichannel (rather than multi z slice)
	:type multichannel_label_image: bool
	:param zstack: if True images are interpreted as zstack (rather than multichannel)
	:type zstack: bool
	'''

	c = TiffConnector(img_filepath,label_filepath, savepath=savepath\
		, multichannel_pixel_image=multichannel_pixel_image\
        , multichannel_label_image=multichannel_label_image\
        , zstack=zstack)
	d = Dataset(c)

	mb = Minibatch(d, minibatch_size, tpl_size_zxy, padding_zxy=padding_zxy)
	pd = PredictionData(d, tpl_size_zxy, padding_zxy=padding_zxy)

	return mb, pd




