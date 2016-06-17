import numpy as np

class Image(object):
    '''
    provides connectors to pixel data source and
    (optionally) assigned weights for classifier training

    provides methods for getting image templates and data 
    augmentation (mini batch) for training

    an image object is loaded into memory, so it should not
    exceed a certain size.

    to handle large images, these should be divided into
    multiple image objects and handled in a training_data class
    (or prediction_data) class
    '''
    

    def __init__(self, importerFunc, importerFuncArgs):
        
        self.importerFunc = importerFunc
        self.importerFuncArgs = importerFuncArgs

        self.pixels = None
        self.dims = None
        
        self.load_pixels()
        


    def __repr__(self):

        infostring = \
            'YAPIC Image object \n' \
            'nr of channels: %s \n' \
            'nr of z slices: %s \n' \
            'width (x): %s \n' \
            'height (y): %s \n' \
             % (self.dims[0]\
                , self.dims[1]\
                , self.dims[2]\
                , self.dims[3])

        return infostring

    
    def load_pixels(self):
        '''
        load complete pixel data of image
        '''
        self.pixels = self.importerFunc(*self.importerFuncArgs)
        self.dims = self.pixels.shape
        return    


    def add_weights(self):
        '''
        weights have same dimensions as pixels
        weights are optional: not required for prediction,
        but required for training.

        weights are floating point values bewtween 0 and 1
        if the training data has on/off 'labels' instead of continuous
        weights, the weight value is either 0 or 1

        '''    






def get_template(image, pos, size, padding=0):
    '''
    returns a recangular subsection of an image with specified size.
    :param image: n dimensional image
    :type image: numpy array
    :param pos: tuple defining the upper left position of the template in n dimensions
    :type pos: tuple
    :param size: tuple defining size of template in all dimensions
    :type size: tuple
    :returns: template as numpy array with same nr of dimensions as image


    '''
    pos = np.array(pos)
    size = np.array(size)

    if len(image.shape) != len(size):        
        error_str = '''nr of image dimensions (%s) 
            and size vector length (%s) do not match'''\
            % (len(image.shape), len(size))
        raise ValueError(error_str)

    if len(image.shape) != len(pos):        
        error_str = '''nr of image dimensions (%s) 
            and pos vector length (%s) do not match'''\
            % (len(image.shape), len(pos))
        raise ValueError(error_str)    

    
    if (image.shape < (pos + size)).any():
        error_str = '''template out of image bounds: image shape: %s,  
            pos: %s, template size: %s''' % (image.shape, pos, size)
        raise ValueError(error_str)


    if padding == 0:    
        indices = get_indices(pos, size)
        return image[np.meshgrid(*indices, indexing='ij')]

    pos_p = pos-padding
    size_p = size + 2*padding
    
    reflect_sizes = get_padding_size(image.shape, pos_p, size_p)
    image = np.pad(image, reflect_sizes, mode='reflect')
    pos_corr = correct_pos_for_padding(pos_p, reflect_sizes)

    indices = get_indices(pos_corr, size_p)
    
    return image[np.meshgrid(*indices, indexing='ij')]






def correct_pos_for_padding(pos, padding_sizes):
    return tuple([s[0]+p for p,s in zip(pos, padding_sizes)])


def get_indices(pos, size):
    '''
    returns all indices for a sub matrix, given a certain n-dimensional position (upper left)
    and n-dimensional size. 


    :param pos: tuple defining the upper left position of the template in n dimensions
    :param size: tuple defining size of template in all dimensions
    :returns: list of indices for all dimensions
    '''
    if len(pos) != len(size):
        error_str = '''nr of dimensions does not fit: pos(%s), 
            size (%s)''' % (pos, size)
        raise ValueError(error_str)


    return [list(range(p, p+s)) for p,s in zip(pos,size)]


def get_padding_size(shape, pos, size):
    padding_size = []
    for sh, po, si in zip(shape, pos, size):
        p_l = 0
        p_u = 0
        if po<0:
            p_l = abs(po)
        if po + si > sh:
            p_u = po + si - sh
        padding_size.append((p_l,p_u))        
    return padding_size
    





def are_all_elements_uneven(vals):
    '''
    check if all elements of an array are uneven
    '''
    return (np.array(vals) % 2 != 0).all()

