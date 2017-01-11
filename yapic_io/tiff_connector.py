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
    image filepath: yapic_io/test_data/tiffconnector_1/im
    label filepath: yapic_io/test_data/tiffconnector_1/labels
    <BLANKLINE>
    '''
    
    def __init__(self\
            , img_filepath, label_filepath, savepath=None\
            , multichannel_pixel_image=None\
            , multichannel_label_image=None\
            , zstack=True):
        
        '''
        :param img_filepath: path to source pixel images (use wildcards for filtering)
                             or a list of filenames
        :param label_filepath: path to label images (use wildcards for filtering)
                               or a list of filenames
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

        if type(img_filepath) == str:
            assert type(label_filepath) == str

            img_filepath   = os.path.normpath(os.path.expanduser(img_filepath))
            label_filepath = os.path.normpath(os.path.expanduser(label_filepath))

            if os.path.isdir(img_filepath):
                img_filepath = os.path.join(img_filepath, '*.tif')
            if os.path.isdir(label_filepath):
                label_filepath = os.path.join(label_filepath, '*.tif')

            self.img_path, img_filemask = os.path.split(img_filepath)
            self.label_path, label_filemask = os.path.split(label_filepath)

            img_filenames = self.load_img_filenames(img_filemask)
            lbl_filenames = self.load_label_filenames(label_filemask)

            self.filenames = find_best_matching_pairs(img_filenames, lbl_filenames)
        else:
            assert type(img_filepath)
            assert type(label_filepath)

            img_filenames = img_filepath
            lbl_filenames = label_filepath

            if len(img_filenames) > 0:
                self.img_path, _ = os.path.split(img_filenames[0])
                img_filenames = [os.path.split(fname)[-1] if fname is not None else None for fname in img_filenames]
            else:
                self.img_path = None

            filtered_labels = [fname for fname in lbl_filenames if fname is not None]
            if len(filtered_labels) > 0:
                self.label_path, _ = os.path.split(filtered_labels[0])
                lbl_filenames = [os.path.split(fname)[-1] if fname is not None else None for fname in lbl_filenames]
            else:
                self.label_path = None

            self.filenames = list(zip(img_filenames, lbl_filenames))

        assert img_filenames is not None
        assert lbl_filenames is not None
        if len(img_filenames) != len(lbl_filenames):
            msg = 'Number of image files ({}) and label files ({}) differ!'
            logger.warning(msg.format(len(img_filenames), len(lbl_filenames)))

        logger.debug('Pixel and label files are assigned as follows:')
        logger.debug('\n'.join('{} <-> {}'.format(img, lbl) for img, lbl in  self.filenames))

        self.savepath = savepath #path for probability maps

        self.check_labelmat_dimensions()
        self.map_labelvalues()
        

    def __repr__(self):
        infostring = \
            'Connector_tiff object\n' \
            'image filepath: {}\n' \
            'label filepath: {}\n'.format(self.img_path, self.label_path)
        return infostring


    def filter_labeled(self):
        '''
        Returns a new TiffConnector containing only images that have labels
        '''
        img_fnames = [os.path.join(self.img_path, img) for img, lbl in self.filenames
                      if lbl is not None]

        lbl_fnames = [os.path.join(self.label_path, lbl) if lbl is not None else None
                      for img, lbl in self.filenames
                      if lbl is not None]

        return TiffConnector(img_fnames, lbl_fnames,
                             savepath=self.savepath,
                             multichannel_pixel_image=self.multichannel_pixel_image,
                             multichannel_label_image=self.multichannel_label_image,
                             zstack=self.zstack)


    def split(self, fraction, random_seed=42):
        '''
        Split the images pseudo-randomly into two subsets (both TiffConnectors).
        The first of size `(1-fraction)*N_images`, the other of size `fraction*N_images`
        '''
        N = len(self.filenames)

        state = np.random.get_state()
        np.random.seed(random_seed)
        mask = np.random.choice([True, False], size=5, p=[1-fraction, fraction])
        np.random.set_state(state)

        img_fnames1 = [os.path.join(self.img_path, img)
                      for m, (img, lbl) in zip(mask, self.filenames) if m == True]
        lbl_fnames1 = [os.path.join(self.label_path, lbl) if lbl is not None else None
                       for m, (img, lbl) in zip(mask, self.filenames) if m == True]

        img_fnames2 = [os.path.join(self.img_path, img)
                      for m, (img, lbl) in zip(mask, self.filenames) if m == False]
        lbl_fnames2 = [os.path.join(self.label_path, lbl) if lbl is not None else None
                       for m, (img, lbl) in zip(mask, self.filenames) if m == False]

        if len(img_fnames1) == 0:
            warining.warn('TiffConnector.split({}): First connector is empty!'.format(fraction))
        if len(img_fnames1) == N:
            warining.warn('TiffConnector.split({}): Second connector is empty!'.format(fraction))

        conn1 = TiffConnector(img_fnames1, lbl_fnames1,
                              savepath=self.savepath,
                              multichannel_pixel_image=self.multichannel_pixel_image,
                              multichannel_label_image=self.multichannel_label_image,
                              zstack=self.zstack)
        conn2 = TiffConnector(img_fnames2, lbl_fnames2,
                              savepath=self.savepath,
                              multichannel_pixel_image=self.multichannel_pixel_image,
                              multichannel_label_image=self.multichannel_label_image,
                              zstack=self.zstack)

        return conn1, conn2



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


    @lru_cache(maxsize=5000)    
    def get_label_template(self, image_nr=None, pos=None, size=None):        
        labelmat = self.load_label_matrix(self, image_nr)
        mesh = get_template_meshgrid(labelmat.shape, pos, size)

        return labelmat[mesh]

    @lru_cache(maxsize=5000)    
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
            raise ValueError          

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
        logger.info('Checking labelmatrix dimensions...')
        nr_channels = []
        for image_nr in list(range(self.get_image_count())):
            im_dim = self.load_img_dimensions(image_nr)
            label_dim = self.load_labelmat_dimensions(image_nr)

            if label_dim is None:
                logger.debug('Check image nr %s: ok (no labelmat found) ', image_nr)
            else:
                nr_channels.append(label_dim[0])
                logger.debug('Found %s label channel(s)', nr_channels[-1])
                
                if label_dim[1:] == im_dim[1:]:
                    logger.debug('Check image nr %s: ok ', image_nr)
                else:
                    logger.error('Check image nr %s (%s): image dim is %s, label dim is %s '\
                        , image_nr, self.filenames[image_nr], im_dim, label_dim)
                    raise ValueError('Check image nr %s: dims do not match ' % str(image_nr))   
        if len(set(nr_channels))>1:
            raise ValueError('Nr of channels not consitent in input data, found following nr of labelmask channels: %s' % str(set(nr_channels))) 

        logger.info('Labelmatrix dimensions ok')               

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


    
    def get_template_for_label(self, image_nr, pos_zxy, size_zxy, label_value):
        '''
        returns a 3d zxy boolean matrix where positions of the reuqested label
        are indicated with True. only mapped labelvalues can be requested.
        '''    

        labelmat = self.load_label_matrix(image_nr) #matrix with labelvalues
        boolmat_4d = (labelmat == label_value)

        boolmat_3d = boolmat_4d.any(axis=0) #reduction to zxy dimension
        #comment: mapped labelvalues are unique for a channel, as they
        #are generated with map_labelvalues(). This means,
        #a mapped labelvalue is only present in one specific channel.
        #This means: there chould be not more than one truthy value along the
        #channel dimension in boolmat_4d. this is not doublechecked here.

        mesh = get_template_meshgrid(boolmat_3d.shape, pos_zxy, size_zxy)

        return boolmat_3d[mesh]






    @lru_cache(maxsize = 20)
    def load_label_matrix(self, image_nr, original_labelvalues=False):

        '''
        returns a 4d labelmatrix with dimensions czxy.
        the albelmatrix consists of zeros (no label) or the respective
        label value.

        if original_labelvalues is False, the mapped label values are returned,
        otherwise the original labelvalues.
        '''
        
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
        logger.info('Mapping label values...')

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

        logger.info('Label values are mapped to ascending values:')
        logger.info(label_mappings)
        return label_mappings           



    def get_original_labelvalues(self):
        '''
        returns a list of sets. each set corresponds to 1 label channel.
        each set contains the label values of that channel.
        '''
        
        labels_per_channel = []
        for image_nr in range(self.get_image_count()):
            labels_per_im = self.get_original_labelvalues_for_im(image_nr)

            if labels_per_im is not None:
                if len(labels_per_channel) == 0:
                    labels_per_channel = labels_per_im
                else:
                    labels_per_channel = [l1.union(l2) for l1, l2
                                          in zip(labels_per_channel, labels_per_im)]
        return labels_per_channel


    @lru_cache(maxsize = 5000)  
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






    @lru_cache(maxsize = 500)    
    def get_labelvalues_for_im(self, image_nr):
        mat = self.load_label_matrix(image_nr)
        if mat is None:
            return None
        values =  np.unique(mat)
        values = values[values!=0]
        values.sort()
        return list(values)

    @lru_cache(maxsize = 500)    
    def get_labelcount_for_im(self, image_nr):
        '''
        returns for each label value the number of labels for this image
        ''' 

        mat = self.load_label_matrix(image_nr)
        labels = self.get_labelvalues_for_im(image_nr)
        if labels is None:
            #if no labelmatrix available
            return None
        
        label_count = {}
        for label in labels:
            label_count[label] = np.count_nonzero(mat==label)
        
        return label_count

    
    def labelvalue_is_valid(self,label_value):
        labelvalues = []
        for d in self.labelvalue_mapping:
            labelvalues += d.values()

        if label_value in set(labelvalues):
            return True
        return False    
            

    def get_label_coordinate(self, image_nr, label_value, label_index):
        '''
        returns a czxy coordinate of a specific label (specified by the
        label index) with labelvalue label_value (mapped label value).

        The count of labels for a specific labelvalue can be retrieved by
        
        count = get_labelcount_for_im()

        The label_index must be a value between 0 and count[label_value].
        '''      
        mat = self.load_label_matrix(image_nr)
        
        #check for correct label_value
        if not self.labelvalue_is_valid(label_value):
            raise ValueError('Label value %s does not exist. Label value mapping: %s' %\
                (str(label_value), str(self.labelvalue_mapping)))

        #label matrix
        mat = self.load_label_matrix(image_nr)

        coors = np.array(np.where(mat==label_value))
        
        n_coors = coors.shape[1]
        if (label_index < 0)  or (label_index >= n_coors):
            raise ValueError('''Label index %s for label value %s in image %s 
                not correct. Only %s labels of that value for this image''' %\
                (str(label_index), str(label_value), str(image_nr), str(n_coors)))

        coor = coors[:,label_index]    
        coor[0] = 0 #set channel to zero
        
        return coor



    


    def is_valid_image_nr(self, image_nr):
        count = self.get_image_count()
        
        error_msg = \
        'wrong image number. image numbers in range 0 to %s' % str(count-1)
        
        if image_nr not in list(range(count)):
            logger.error(error_msg)
            return False

        return True          


    def load_label_filenames(self, filemask):
        label_filenames = sorted(glob.glob(os.path.join(self.label_path, filemask)))
        label_filenames = [os.path.split(fname)[1] for fname in label_filenames]
        return label_filenames


    def load_img_filenames(self, filemask):
        '''
        find all tiff images in specified folder (self.img_path, filemask)
        '''
        filenames = sorted(glob.glob(os.path.join(self.img_path, filemask)))
        filenames = [os.path.split(fname)[1] for fname in filenames]
        
        logger.info('{} pixel image files detected.'.format(len(filenames)))
        logger.debug('Pixel image files:')
        logger.debug(filenames)
        return filenames

