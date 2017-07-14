from yapic_io.tiff_connector import TiffConnector
from yapic_io.coordinate_connector import CoordinateConnector
import numpy as np


class CoordinateTiffConnector(TiffConnector, CoordinateConnector):
    '''
    extends tiff connector with label_to_index_coordinate function.

    effectively, this will change the training tile fetching from
    polling to fetching by_label
    '''

    def __init__(self,
                 img_filepath, label_filepath, savepath=None,
                 multichannel_pixel_image=None,
                 multichannel_label_image=None,
                 zstack=True):
        super().__init__(img_filepath, label_filepath,
                         savepath=savepath,
                         multichannel_pixel_image=multichannel_pixel_image,
                         multichannel_label_image=multichannel_label_image,
                         zstack=zstack)

    def label_index_to_coordinate(self, image_nr, label_value, label_index):
        '''
        returns a czxy coordinate of a specific label (specified by the
        label index) with labelvalue label_value (mapped label value).

        The count of labels for a specific labelvalue can be retrieved by

        count = label_count_for_image()

        The label_index must be a value between 0 and count[label_value].
        '''
        mat = self.load_label_matrix(image_nr)

        # check for correct label_value
        if not self.is_labelvalue_valid(label_value):
            raise ValueError('Label value %s does not exist. Label value mapping: %s' %
                             (str(label_value), str(self.labelvalue_mapping)))

        # label matrix
        mat = self.load_label_matrix(image_nr)

        coors = np.array(np.where(mat.ravel() == label_value))

        n_coors = coors.size
        if (label_index < 0) or (label_index >= n_coors):
            raise ValueError('''Label index %s for label value %s in image %s
                not correct. Only %s labels of that value for this image''' %
                             (str(label_index), str(label_value), str(image_nr), str(n_coors)))

        coor = np.unravel_index(coors[0, label_index], mat.shape)
        coor = np.array(coor)
        coor[0] = 0  # set channel to zero

        return coor
