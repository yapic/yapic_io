from yapic_io.tiff_connector import TiffConnector
from yapic_io.dataset import Dataset
from yapic_io.prediction_data import PredictionData
from yapic_io.minibatch import Minibatch


def make_tiff_interface(img_filepath, label_filepath, savepath,\
		tpl_size_zxy, minibatch_size=10, padding_zxy=(0,0,0)):
	
	'''
	Convenience method to make a Minibatch and a PredictionData
	object bound to data from a TiffConnector
	'''

	c = TiffConnector(img_filepath,label_filepath, savepath=savepath)
	d = Dataset(c)

	mb = Minibatch(d, minibatch_size, tpl_size_zxy, padding_zxy=padding_zxy)
	pd = PredictionData(d, tpl_size_zxy, padding_zxy=padding_zxy)

	return mb, pd




