import os
import numpy as np



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

def get_template_meshgrid(image_shape, pos, size):
    '''
    returns coordinates for selection of a sub matrix
    , given a certain n-dimensional position (upper left)
    and n-dimensional size. 

    :param shape: tuple defining the image shape
    :param pos: tuple defining the upper left position of the template in n dimensions
    :param size: tuple defining size of template in all dimensions
    :returns: a multidimensional meshgrid defining the sub matrix

    '''

    pos = np.array(pos)
    size = np.array(size)

    if len(image_shape) != len(size):        
        error_str = '''nr of image dimensions (%s) 
            and size vector length (%s) do not match'''\
            % (len(image_shape), len(size))
        raise ValueError(error_str)

    if len(image_shape) != len(pos):        
        error_str = '''nr of image dimensions (%s) 
            and pos vector length (%s) do not match'''\
            % (len(image_shape), len(pos))
        raise ValueError(error_str)    

    
    if (image_shape < (pos + size)).any():
        error_str = '''template out of image bounds: image shape: %s,  
            pos: %s, template size: %s''' % (image_shape, pos, size)
        raise ValueError(error_str)

    indices = get_indices(pos, size)
    return np.meshgrid(*indices, indexing='ij') 






def filterList(inp,pattern,mode='include'):
    out = []
    
    #print inp, 'pattern=', pattern
    if mode =='include':
        for el in inp:
            if el.find(pattern) != -1:
                out.append(el)
    if mode == 'avoid':
        for el in inp:
            if el.find(pattern) == -1:
                out.append(el)

    return out      




def getFilelistFromDir(folder,pattern,avoidpattern=None):
    """Returns a list of files from a given source folder. 
    Filenames must match the list of patterns and must not include
    the optional list of avoidpatterns 

    :param folder: root folder where files are located
    :type folder: string
    :param pattern: string patterns. each filename in the output filename list matches all patterns 
    :param avoidpattern: string patterns. each filename in the output list does not match any of the avoidpatterns
    :type pattern: list of strings
    :type avoidpattern: list of strings
    :rtype: list of strings
    """
     

    allfiles = os.listdir(folder)
    selectedfiles = []

    pt = my_toList(pattern) #convert pattern string to list
    flist = allfiles    
    for patel in pt:
        flist =  filterList(flist, patel)
    

    if avoidpattern:
        av = my_toList(avoidpattern)
        for patel in av:
            flist = filterList(flist, patel, mode = 'avoid')
    return flist        


def my_toList(pattern):
    if type(pattern) is str: # if only one pattern is available
        return [pattern]
    else:
        return pattern  
