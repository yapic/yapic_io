from abc import ABCMeta, abstractmethod
from yapic_io.connector import Connector

class Coordinate_connector(Connector, metaclass=ABCMeta):
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

    def __init__(self):
        self.tile_fetching_mode = 'by_label_index'


    @abstractmethod
    def label_index_to_coordinate(self, image_nr, label_value, label_index):
        '''
        returns a czxy coordinate of a specific label (specified by the
        label index) with labelvalue label_value (mapped label value).
        
        The count of labels for a specific labelvalue can be retrieved by
        
        count = label_count_for_image()
        
        The label_index must be a value between 0 and count[label_value].
        '''
        
        pass