import logging
import os
import glob
import itertools
from functools import lru_cache
import yapic_io.utils as ut
import numpy as np

import yapic_io.image_importers as ip
from yapic_io.utils import get_template_meshgrid, add_to_filename, find_best_matching_pairs
from yapic_io.connector import Connector
from pprint import pprint
logger = logging.getLogger(os.path.basename(__file__))


class TiffConnector(Connector):
    '''
    implementation of Connector for normal sized tiff images (up to 4 dimensions)
    and corresponding label masks (up to 4 dimensions) in tiff format.
    
    Initiate a new TiffConnector as follows:

    >>> from yapic_io.tiff_connector import TiffConnector
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> t = TiffConnector(pixel_image_dir, label_image_dir)
    >>> print(t)
    Connector_tiff object
    image filepath: yapic_io/test_data/tiffconnector_1/im/*.tif
    label filepath: yapic_io/test_data/tiffconnector_1/labels/*.tif
    <BLANKLINE>
    '''
    def __init__(self\
            , img_filepath, label_filepath, savepath=None\
            , multichannel_pixel_image=None\
            , multichannel_label_image=None\
            , zstack=True):
        
        '''
        :param img_filepath: path to source pixel images, use wildcards for filtering
        :param label_filepath: path to label images, use wildcards for filtering
        :param savepath: path for output probability images
        :param multichannel_pixel_image: set True if pixel images have multiple channels
        :type multichannel_pixel_image: bool
        :param multichannel_label_image: set True if label images have multiple channels
        :type multichannel_label_image: bool
        :param zstack: set True if label- and pixel images are zstacks
        :type zstack: bool

        Label images and pixel images have to be equal in zxy dimensions, but can differ
        in nr of channels.

        Labels can be read from multichannel images. This is needed for networks
        with multple output layers. Each channel is assigned one output layer.
        Different labels from different channels can overlap (can share identical
        xyz positions).

        Multichannel_pixel_image,  multichannel_pixel_image and zstack
        can be set to None. In this case the importer tries to map 
        dimensions automatically. This does not always work, esp. in case
        of 3 dimensional images. 

        
        Examples:
        
        - If zstack is set to False and multichannel_pixel_image is set to None,
          the importer will assign the thrid dimensions (in case of 3 dimensional images)
          to channels, i.e. interprets the image as multichannel, single z image.

        - If zstack is set to None and multichannel_pixel_image is set to None,
          the importer will assign all dims correctly in case of 4 dimensional images
          and in case of 2 dimensional images (single z, singechannel). In case of 3
          dimensional images, it throws an error, because it is not clear if the thrid
          dimension is z or channel (RGB images will still be mapped correctly) 


        '''

        self.filenames = None # list of tuples: [(imgfile_1.tif, labelfile_1.tif), (imgfile_2.tif, labelfile_2.tif), ...] 

        self.labelvalue_mapping = None #list of dicts of original and assigned labelvalues
        
        #reason: assign unique labelvalues
        # [{orig_label1: 1, orig_label2: 2}, {orig_label1: 3, orig_label2: 4}, ...]

        self.zstack = zstack
        self.multichannel_pixel_image = multichannel_pixel_image
        self.multichannel_label_image = multichannel_label_image



        img_filepath   = os.path.normpath(os.path.expanduser(img_filepath))
        label_filepath = os.path.normpath(os.path.expanduser(label_filepath))

        
        

        if os.path.isdir(img_filepath):
            img_filepath = os.path.join(img_filepath, '*.tif')
        if os.path.isdir(label_filepath):
            label_filepath = os.path.join(label_filepath, '*.tif')

        self.img_path, self.img_filemask = os.path.split(img_filepath)
        self.label_path, self.label_filemask = os.path.split(label_filepath)
        self.savepath = savepath #path for probability maps

        self.load_img_filenames()
        self.load_label_filenames()
        
        self.check_labelmat_dimensions()
        self.map_labelvalues()
        

    def __repr__(self):
        infostring = \
            'Connector_tiff object\n' \
            'image filepath: %s\n' \
            'label filepath: %s\n' \
             % (os.path.join(self.img_path, self.img_filemask)\
                , os.path.join(self.label_path, self.label_filemask))

        return infostring


    def get_image_count(self):
        if self.filenames is None:
            return 0
        
        return len(self.filenames)


    def put_template(self, pixels, pos_zxy, image_nr, label_value):
        if not len(pos_zxy) == 3:
            raise ValueError('pos_zxy has not length 3: %s' % str(pos_zxy))

        if not len(pixels.shape) == 3:
            raise ValueError('''probability map pixel template
             must have 3 dimensions (z,x,y), but has %s : 
             pixels shape is %s''' % \
             (str(len(pixels.shape)), str(pixels.shape)))

        out_path = self.init_probmap_image(image_nr, label_value)

        logger.info('try to add new pixels to  image %s', out_path)
        return ip.add_vals_to_tiff_image(out_path, pos_zxy, pixels)


    def init_probmap_image(self, image_nr, label_value, overwrite=False):
        out_path = self.get_probmap_path(image_nr, label_value)
        _, z_shape, x_shape, y_shape = self.load_img_dimensions(image_nr)
        
        if not os.path.isfile(out_path) or overwrite:
            ip.init_empty_tiff_image(out_path, x_shape, y_shape, z_size=z_shape)
            logger.info('initialize a new probmap image: %s', out_path)
        return out_path        

    def get_probmap_path(self, image_nr, label_value):
        if self.savepath is None:
            raise ValueError('savepath not set')
        image_filename = self.filenames[image_nr][0]
        probmap_filename = add_to_filename(image_filename,\
                     'class_' + str(label_value))
        return os.path.join(self.savepath, probmap_filename)


            

    def get_template(self, image_nr=None, pos=None, size=None):

        im = self.load_image(image_nr)
        mesh = get_template_meshgrid(im.shape, pos, size)

        return(im[mesh])


    def load_img_dimensions(self, image_nr):
        '''
        returns dimensions of the dataset.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''        

        if not self.is_valid_image_nr(image_nr):
            return False          

        path = os.path.join(self.img_path, self.filenames[image_nr][0])
        return ip.get_tiff_image_dimensions(path,\
            zstack=self.zstack, multichannel=self.multichannel_pixel_image) 

    def load_labelmat_dimensions(self, image_nr):
        '''
        returns dimensions of the label image.
        dims is a 4-element-tuple:
        
        :param image_nr: index of image
        :returns (nr_channels, nr_zslices, nr_x, nr_y)

        '''        

        if not self.is_valid_image_nr(image_nr):
            return False          

        if self.exists_label_for_img(image_nr):
            path = os.path.join(self.label_path, self.filenames[image_nr][1])
            return ip.get_tiff_image_dimensions(path,\
                zstack=self.zstack, multichannel=self.multichannel_label_image)               
    

    def check_labelmat_dimensions(self):
        '''
        check if label mat dimensions fit to image dimensions, i.e.
        everything identical except nr of channels (label mat always 1)
        '''
        logger.info('checking labelmatrix dimensions...')
        nr_channels = []
        for image_nr in list(range(self.get_image_count())):
            im_dim = self.load_img_dimensions(image_nr)
            label_dim = self.load_labelmat_dimensions(image_nr)

            if label_dim is None:
                logger.debug('check image nr %s: ok (no labelmat found) ', image_nr)
            else:
                nr_channels.append(label_dim[0])
                logger.info('found %s label channel(s)', nr_channels[-1])
                
                if label_dim[1:] == im_dim[1:]:
                    logger.info('check image nr %s: ok ', image_nr)
                else:
                    logger.error('check image nr %s (%s): image dim is %s, label dim is %s '\
                        , image_nr, self.filenames[image_nr], im_dim, label_dim)
                    raise ValueError('check image nr %s: dims do not match ' % str(image_nr))   
        if len(set(nr_channels))>1:
            raise ValueError('nr of channels not consitent in input data, found following nr of labelmask channels: %s' % str(set(nr_channels))) 

        logger.info('labelmatrix dimensions ok')               

    @lru_cache(maxsize = 20)
    def load_image(self, image_nr):
        if not self.is_valid_image_nr(image_nr):
            return None      
        path = os.path.join(self.img_path, self.filenames[image_nr][0])
        return ip.import_tiff_image(path)    


    def exists_label_for_img(self, image_nr):
        if not self.is_valid_image_nr(image_nr):
            return None      
        if self.filenames[image_nr][1] is None:
            return False
        return True    


    
    def load_label_matrix(self, image_nr, original_labelvalues=False):
        
        if not self.is_valid_image_nr(image_nr):
            return None      

        label_filename = self.filenames[image_nr][1]      
        
        if label_filename is None:
            logger.warning('no label matrix file found for image file %s', str(image_nr))    
            return None
        
        path = os.path.join(self.label_path, label_filename)
        logger.debug('try loading labelmat %s',path)
        
        label_image = ip.import_tiff_image(path,\
                zstack=self.zstack, multichannel=self.multichannel_label_image)    

        if original_labelvalues:
            return label_image
        
        label_image = ut.assign_slice_by_slice(self.labelvalue_mapping, label_image) 
        
        return label_image 


    
    def map_labelvalues(self):
        '''
        assign unique labelvalues to original labelvalues.
        for multichannel label images it might happen, that identical
        labels occur in different channels.
        to avoid conflicts, original labelvalues are mapped to unique values
        in ascending order 1,2,3,4...

        This is defined in self.labelvalue_mapping:

        [{orig_label1: 1, orig_label2: 2}, {orig_label1: 3, orig_label2: 4}, ...]

        Each element of the list correponds to one label channel.
        Keys are the original labels, values are the assigned labels that
        will be seen by the Dataset object.
        '''
        logger.info('mapping labelvalues...')

        label_mappings = []
        o_labelvals = self.get_original_labelvalues()
        new_label = 1
        for labels_per_channel in o_labelvals:
            label_mapping = {}
            labels_per_channel = sorted(list(labels_per_channel))
            for label in labels_per_channel:
                label_mapping[label] = new_label
                new_label += 1
            label_mappings.append(label_mapping)
        
        self.labelvalue_mapping = label_mappings

        logger.info('label values are mapped to ascending values:')
        logger.info(label_mappings)
        return label_mappings           




    def get_original_labelvalues(self):
        '''
        returns a list of sets. each set corresponds to 1 label channel.
        each set contains the label values of that channel.
        '''
        
        label_list_of_lists = []
        for image_nr in range(self.get_image_count()):
            labels_per_im = self.get_original_labelvalues_for_im(image_nr)
            if labels_per_im is not None:
                label_list_of_lists.append(labels_per_im)
        if len(label_list_of_lists)==0:
            return []

        out = label_list_of_lists[0]
        for labels_per_im in label_list_of_lists:
            for channel, outchannel in zip(labels_per_im, out):
                outchannel = outchannel.union(channel)
        return out        



    def get_original_labelvalues_for_im(self, image_nr):
        mat = self.load_label_matrix(image_nr, original_labelvalues=True)
        if mat is None:
            return None
        

        out = []    
        nr_channels = mat.shape[0]
        for channel in range(nr_channels):
            values =  np.unique(mat[channel,:,:,:])
            values = values[values!=0]
            out.append(set(values))
        return out







    def get_labelvalues_for_im(self, image_nr):
        mat = self.load_label_matrix(image_nr)
        if mat is None:
            return None
        values =  np.unique(mat)
        values = values[values!=0]
        values.sort()
        return list(values)


    @lru_cache(maxsize = 500)
    def get_label_coordinates(self, image_nr):
        ''''
        returns label coordinates as dict in following format:
        channel has always value 0!! This value is just kept for consitency in 
        dimensions with corresponding pixel data 

        {
            label_nr1 : [(channel, z, x, y), (channel, z, x, y) ...],
            label_nr2 : [(channel, z, x, y), (channel, z, x, y) ...],
            ...
        }


        :param image_nr: index of image
        :returns: dict of label coordinates
    
        '''

        mat = self.load_label_matrix(image_nr)
        labels = self.get_labelvalues_for_im(image_nr)
        if labels is None:
            #if no labelmatrix available
            return None
        
        label_coor = {}
        for label in labels:
            coors = np.array(np.where(mat==label))
            
            
            coor_list = [tuple(coors[:,i]) for i in list(range(coors.shape[1]))]
            coor_list = [(0,) + c[1:] for c in coor_list] #set channel always to 1
            label_coor[label] = coor_list#np.array(np.where(mat==label))
        
        return label_coor    


    def is_valid_image_nr(self, image_nr):
        count = self.get_image_count()
        
        error_msg = \
        'wrong image number. image numbers in range 0 to %s' % str(count-1)
        
        if image_nr not in list(range(count)):
            logger.error(error_msg)
            return False

        return True          


    def load_label_filenames(self):
        if self.filenames is None:
            return

        # if mode is None:
        #     if os.path.isdir(self.label_path) and os.path.samefile(self.label_path, self.img_path):
        #         mode = 'order'
        #     else:
        #         mode = 'identical'

        # if mode == 'identical':
        #     #if label filenames should match exactly with img filenames
        #     for name_tuple in self.filenames:
        #         img_filename = name_tuple[0]   
        #         label_path = os.path.join(self.label_path, img_filename)
        #         if os.path.isfile(label_path): #if label file exists for image file
        #             name_tuple[1] = img_filename
        #elif mode == 'order':
        image_filenames = [pair[0] for pair in self.filenames]
        label_filenames = sorted(glob.glob(os.path.join(self.label_path, self.label_filemask)))
        label_filenames = [os.path.split(fname)[1] for fname in label_filenames]

        if len(image_filenames) != len(label_filenames):
            msg = 'Number of image files ({}) and label files ({}) differ!'
            logger.info(msg.format(len(image_filenames), len(label_filenames)))

        self.filenames = find_best_matching_pairs(image_filenames, label_filenames)

        logger.info('pixel and label files are assigned as follows:')
        logger.info(self.filenames)

            #self.filenames = list(zip(image_filenames, label_filenames))
        #else:
        #    raise NotImplemented


    # def check_filename_similarity(self):
    #     distances = list(itertools.starmap(lev_distance, self.filenames))
    #     dist_min = np.amin(distances)
    #     dist_argmax = np.argmax(distances)
    #     dist_max = distances[dist_argmax]

    #     if dist_max - dist_min > 0:
    #         logger.warn('Odd filename pair detected {}'.format(self.filenames[dist_argmax]))
    #         return False
    #     return True


    def load_img_filenames(self):
        '''
        find all tiff images in specified folder (self.img_path, self.img_filemask)
        '''
        img_filenames = sorted(glob.glob(os.path.join(self.img_path, self.img_filemask)))
        
        filenames = [[os.path.split(filename)[1], None] for filename in img_filenames]
        if len(filenames) == 0:
            self.filenames = None
            return
        self.filenames = filenames
        
        logger.info('following pixel image files detected:')
        logger.info(img_filenames)
        return True 

