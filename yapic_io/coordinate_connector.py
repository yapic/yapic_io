from abc import ABCMeta, abstractmethod
from yapic_io.connector import Connector


class CoordinateConnector(Connector, metaclass=ABCMeta):
    '''
    Interface to pixel and label data source for classifier
    training and prediction.
    The connector is optimized for data sources where labels are stored
    in coordinates.

    The Coordinate_connector class has the 'by_label_index' mode to
    fetch training tiles. In this mode a dataset object fetches pixel
    and label tiles by a specific label coordinate.
    This is potentially more efficient than random polling in the following
    situation:

    - very large pixel images (e.g. slide scanner data)
    - sparse labelling where labels are stored as coordinates
    '''

    @abstractmethod
    def label_index_to_coordinate(self, image_nr, label_value, label_index):
        '''
        Get image coordinate for specific label.

        Parameters
        ----------
        image_nr : int
            Index of image.
        label_value : int
            Id of the label.
        label_index: int
            Value between 0 and count[label_value].
            Label count can be retrieved with self.label_count_for_image
            method.

        Returns
        -------
        ndarray
            czxy coordinate of a specific label (specified by the
            label index) with labelvalue label_value (mapped label value).
        '''

        pass
