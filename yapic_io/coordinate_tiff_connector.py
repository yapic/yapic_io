from yapic_io.tiff_connector import TiffConnector
from yapic_io.coordinate_connector import CoordinateConnector
import yapic_io.utils as ut
import numpy as np


class CoordinateTiffConnector(TiffConnector, CoordinateConnector):
    '''
    extends tiff connector with label_to_index_coordinate function.

    effectively, this will change the training tile fetching from
    polling to fetching by_label
    '''
    def label_index_to_coordinate(self, image_nr, label_value, label_index):
        '''
        returns a czxy coordinate of a specific label (specified by the
        label index) with labelvalue label_value (mapped label value).

        The count of labels for a specific labelvalue can be retrieved by

        count = label_count_for_image()

        The label_index must be a value between 0 and count[label_value].
        '''
        mat = self.load_label_matrix(image_nr)

        valid_labels = ut.flatten(d.values() for d in self.labelvalue_mapping)
        msg = 'Label {} non-existing in label value mapping {}'
        assert label_value in valid_labels, msg.format(label_value, self.labelvalue_mapping)

        coors = np.array(np.where(mat.ravel() == label_value))

        err = 'label index out of range for label_value {}, img {}'.format(label_value, image_nr)
        np.testing.assert_array_less(-1, label_index, coors.size, err)
        np.testing.assert_array_less(label_index, coors.size, err)

        coor = np.unravel_index(coors[0, label_index], mat.shape)
        coor = np.array(coor)
        coor[0] = 0  # set channel to zero

        return coor
