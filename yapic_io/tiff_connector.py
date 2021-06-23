import logging
import os
import collections
from functools import lru_cache
import yapic_io.utils as ut
import numpy as np
import itertools
import warnings
from itertools import zip_longest
from pathlib import Path
from yapic_io.connector import Connector

from tifffile import memmap, TiffFile

logger = logging.getLogger(os.path.basename(__file__))

FilePair = collections.namedtuple('FilePair', ['img', 'lbl'])


def _handle_img_filenames(img_filepath):
    '''
    - checks if list of image filepaths, a single wildcard filepath
      or a single filepath without a wildcard is given.
    - checks if given filenames exit
    - splits into folder and list of filenames
    '''

    if type(img_filepath) in (str, Path):
        img_filepath = os.path.abspath(img_filepath)
        img_filemask = '*.tif' if os.path.isdir(img_filepath)\
            else os.path.basename(img_filepath)

        folder = Path(img_filepath if os.path.isdir(img_filepath)
                      else os.path.dirname(img_filepath))
        filenames = [fname.name for fname in sorted(folder.glob(img_filemask))]

    elif type(img_filepath) in (list, tuple):

        img_filenames = img_filepath
        img_filenames = [Path(p).expanduser().resolve()
                         if p is not None else None
                         for p in img_filepath]

        assert len(img_filenames) > 0, 'list of image filenames is empty'

        for e in img_filenames:
            if e is not None:
                assert e.exists(), 'file {} not found'.format(e)

        folders = {fname.parent
                   for fname in img_filenames if fname is not None}
        assert len(folders) == 1, 'image filenames are not in the same folder'
        folder = next(iter(folders))
        folder = folder.expanduser().resolve()
        filenames = [fname.name
                     if fname is not None else None
                     for fname in img_filenames]

    else:
        raise NotImplementedError(
            'could not import images from {}'.format(img_filepath))

    logger.info('{} image files detected.'.format(len(filenames)))
    return folder, filenames


class TiffConnector(Connector):
    '''
    Implementation of Connector for tiff images up to 4 dimensions and
    corresponding label masks up to 4 dimensions in tiff format.

    Parameters
    ----------
    img_filepath : str or list of str
        Path to source pixel images (use wildcards for filtering)
        or a list of filenames.
    label_filepath : str or list of str
        Path to label images (use wildcards for filtering)
        or a list of filenames.
    savepath : str, optional
        Directory to save pixel classifiaction results as probability
        images.

    Notes
    -----
    Label images and pixel images have to be equal in zxy dimensions,
    but can differ in nr of channels.

    Labels can be read from multichannel images. This is needed for
    networks with multiple output layers. Each channel is assigned one
    output layer. Different labels from different channels can overlap
    (can share identical xyz positions).

    Examples
    --------
    Create a TiffConnector object with pixel and label data.

    >>> from yapic_io.tiff_connector import TiffConnector
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> t = TiffConnector(pixel_image_dir, label_image_dir)
    >>> print(t)
    TiffConnector object
    image filepath: yapic_io/test_data/tiffconnector_1/im
    label filepath: yapic_io/test_data/tiffconnector_1/labels
    nr of images: 3
    labelvalue_mapping: [{91: 1, 109: 2, 150: 3}]

    See Also
    --------
    yapic_io.ilastik_connector.IlastikConnector
    '''

    def __init__(self, img_filepath, label_filepath, savepath=None):

        self.img_path, img_filenames = _handle_img_filenames(img_filepath)
        self.label_path, lbl_filenames = self._handle_lbl_filenames(
            label_filepath)

        assert img_filenames is not None, 'no filenames for pixel images found'
        assert len(img_filenames) != 0, 'no filenames for pixel images found'

        if lbl_filenames is None or len(lbl_filenames) == 0:
            pairs = [(img, None) for img in img_filenames]
        else:
            pairs = ut.find_best_matching_pairs(img_filenames, lbl_filenames)

        self._assemble_filenames(pairs)

        logger.info('Pixel and label files are assigned as follows:')
        logger.info('\n'.join('{p.img} <-> {p.lbl}'.format(p=pair)
                              for pair in self.filenames))

        self.savepath = Path(savepath) if savepath is not None else None

        original_labels = self.original_label_values_for_all_images()
        self.labelvalue_mapping = self.calc_label_values_mapping(
            original_labels)

        self.check_label_matrix_dimensions()

    def _assemble_filenames(self, pairs):
        self.filenames = [FilePair(Path(img), Path(lbl) if lbl else None)
                          for img, lbl in pairs]

    def _handle_lbl_filenames(self, label_filepath):
        return _handle_img_filenames(label_filepath)

    def __repr__(self):

        infostring = \
            'TiffConnector object\n' \
            'image filepath: {}\n' \
            'label filepath: {}\n'\
            'nr of images: {}\n'\
            'labelvalue_mapping: {}'.format(self.img_path,
                                            self.label_path,
                                            self.image_count(),
                                            self.labelvalue_mapping)
        return infostring

    def filter_labeled(self):
        '''
        Removes images without labels.

        Returns
        -------
        TiffConnector
            Connector object containing only images with labels.
        '''
        img_fnames = [self.img_path / img for img, lbl in self.filenames
                      if lbl is not None]

        lbl_fnames = [self.label_path / lbl
                      for img, lbl in self.filenames
                      if lbl is not None]

        return TiffConnector(img_fnames, lbl_fnames, savepath=self.savepath)

    def _split_img_fnames(self, fraction, random_seed=42):
        # i took this out from the split method to be used in split method
        # of child methods (e.g. IlasikConnector)
        N = len(self.filenames)

        state = np.random.get_state()
        np.random.seed(random_seed)
        mask = np.random.choice([True, False], size=N, p=[
                                1 - fraction, fraction])
        np.random.set_state(state)

        img_fnames1 = [self.img_path / img
                       for img, lbl in itertools.compress(self.filenames,
                                                          mask)]

        img_fnames2 = [self.img_path / img
                       for img, lbl in itertools.compress(self.filenames,
                                                          ~mask)]

        if len(img_fnames1) == 0:
            msg = ('TiffConnector.split({}): ' +
                   'First connector is empty!').format(fraction)
            warnings.warn(msg)

        if len(img_fnames2) == 0:
            msg = ('TiffConnector.split({}): ' +
                   'Second connector is empty!').format(fraction)
            warnings.warn(msg)

        return img_fnames1, img_fnames2, mask

    def split(self, fraction, random_seed=42):
        '''
        Split the images pseudo-randomly into two Connector subsets.

        The first of size `(1-fraction)*N_images`, the other of size
        `fraction*N_images`

        Parameters
        ----------
        fraction : float
        random_seed : float, optional

        Returns
        -------
        connector_1, connector_2
        '''

        img_fnames1, img_fnames2, mask = self._split_img_fnames(
            fraction,
            random_seed=random_seed)

        lbl_fnames1 = [self.label_path / lbl if lbl is not None else None
                       for img, lbl in itertools.compress(self.filenames,
                                                          mask)]
        lbl_fnames2 = [self.label_path / lbl if lbl is not None else None
                       for img, lbl in itertools.compress(self.filenames,
                                                          ~mask)]

        conn1 = TiffConnector(img_fnames1, lbl_fnames1, savepath=self.savepath)
        conn2 = TiffConnector(img_fnames2, lbl_fnames2, savepath=self.savepath)

        # ensures that both resulting tiff_connectors have the same
        # labelvalue mapping (issue #1)
        conn1.labelvalue_mapping = self.labelvalue_mapping
        conn2.labelvalue_mapping = self.labelvalue_mapping

        # np.random.seed(None)
        return conn1, conn2

    def image_count(self):
        return len(self.filenames)

    @lru_cache(maxsize=10)
    def _open_probability_map_file(self,
                                   image_nr,
                                   label_value,
                                   multichannel=False):
        # memmap is slow, so we must cache it to be fast!
        fname = self.filenames[image_nr].img
        if multichannel:
            fname = Path('{}.tif'.format(fname.stem))
            n_classes = multichannel
            C = n_classes
        else:
            fname = Path('{}_class_{}.tif'.format(fname.stem, label_value))
            C = 1  # channel in output probmap

        path = self.savepath / fname

        if not path.exists():  # created "empty" tif of shape
            _, Z, X, Y = self.image_dimensions(image_nr)
            return memmap(path, shape=(Z, Y, X, C), dtype='float32')

        return memmap(path)

    def put_tile(self,
                 pixels,
                 pos_zxy,
                 image_nr,
                 label_value,
                 multichannel=False):

        assert self.savepath is not None
        np.testing.assert_equal(len(pos_zxy), 3)
        np.testing.assert_equal(len(pixels.shape), 3)
        pixels = np.array(pixels, dtype=np.float32)

        slices = self._open_probability_map_file(image_nr,
                                                 label_value,
                                                 multichannel=multichannel)

        C = 0
        if multichannel:
            C = label_value - 1
        Z, X, Y = pos_zxy
        ZZ, XX, YY = np.array(pos_zxy) + pixels.shape

        # Temporal dimension shift in pixels
        # current is z, x, y but we use z, y, x
        pixels = np.moveaxis(pixels, (0, 1, 2), (0, 2, 1))

        slices[Z: ZZ, Y: YY, X: XX, C] = pixels

    @staticmethod
    def fix_dims(memmap_array, path):
        """This method will fix the dimensions of memmap_array
        in the order (Z, Y, X, C).
        It will increment the dimensions if memmap_array has less than
        4 dimensions.
        It uses the tiff file metadata to guess the original dimensions
        order and which dimensions are missing.

        Parameters
        ----------
        memmap_array - numpy.memmap: sorted image data array in the dimension
        order (Z, Y, X, C)
        path - string: Tiff file path

        Returns
        -------
        connector_1, connector_2
        """
        # target dims (Z,Y,X,C)
        with TiffFile(path) as tif:
            axes = tif.series[0].axes

        # Adding the missed axis
        dims_dict = {'T': 'Z', 'S': 'C', 'Q': 'C'}
        # The letter to represent each dimension may change depending on the
        # file generation.
        # We will transform this representation to Z, Y, X, C to generalize
        # the process
        axes = axes.translate(axes.maketrans(dims_dict))
        if 'C' not in axes:
            memmap_array = np.expand_dims(memmap_array, axis=-1)
            axes += 'C'
        if 'Z' not in axes:
            memmap_array = np.expand_dims(memmap_array, axis=0)
            axes = 'Z' + axes

        # Sorting the axes
        dim_map = ['ZYXC'.index(dim) for dim in axes]
        memmap_array = np.moveaxis(memmap_array, (0, 1, 2, 3), dim_map)
        return memmap_array

    @lru_cache(maxsize=10)
    def _open_image_file(self, image_nr):
        """Returns memmap object with shape: z, y, x, c"""
        # memmap is slow, so we must cache it to be fast!
        path = self.img_path / self.filenames[image_nr].img
        im_data = memmap(path)
        return self.fix_dims(im_data, path)  # shape order: z, y, x, c

    def image_dimensions(self, image_nr):
        """returns a tuple representing the size of the image in the
        order of: C, Z, X, Y"""
        img = self._open_image_file(image_nr)
        Z, Y, X, C = img.shape
        return (C, Z, X, Y)

    def label_matrix_dimensions(self, image_nr):
        '''
        Get dimensions of the label image.


        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        (nr_channels, nr_zslices, nr_x, nr_y)
            Labelmatrix shape.
        '''
        lbl = self._open_label_file(image_nr)
        if lbl is None:
            return

        Z, Y, X, C = lbl.shape
        return (C, Z, X, Y)

    def check_label_matrix_dimensions(self):
        '''
        Check if label matrix dimensions fit to image dimensions, i.e.
        everything identical except nr of channels (label mat always 1).

        Raises
        ------
        AssertionError
            If label matrix dimensions don't fit to image dimensions.
        '''
        N_channels = None

        for i, (img_fname, lbl_fname) in enumerate(self.filenames):
            img_dim = self.image_dimensions(i)
            lbl_dim = self.label_matrix_dimensions(i)

            msg = 'Dimensions for image #{}: img.shape={}, lbl.shape={}'
            logger.debug(msg.format(i, img_dim, lbl_dim))

            if lbl_dim is None:
                continue

            _,  *img_dim = img_dim
            ch, *lbl_dim = lbl_dim

            if N_channels is None:
                N_channels = ch

            msg = 'Label channels inconsistent for {}'.format(lbl_fname)
            np.testing.assert_equal(N_channels, ch, msg)
            msg = 'Invalid image dims for {} and {}'.format(img_fname,
                                                            lbl_fname)
            np.testing.assert_array_equal(lbl_dim, img_dim, msg)

    def _mapped_label_value_to_original(self, label_value):
        '''
        self.labelvalue_mapping in reverse
        '''
        for c, mapping in enumerate(self.labelvalue_mapping):
            reverse_mapping = {v: k for k, v in mapping.items()}
            original = reverse_mapping.get(label_value)
            if original is not None:
                return c, original

        msg = 'Should not be reached! (mapped_label_value={}, mapping={})'
        raise Exception(msg.format(label_value, self.labelvalue_mapping))

    def get_tile(self, image_nr, pos, size):
        """Returns a tile of image as a numpy array.
        the output shape corresponds to: c, z, x, y"""
        C, Z, X, Y = pos
        CC, ZZ, XX, YY = np.array(pos) + size

        slices = self._open_image_file(image_nr)

        tile = np.array(slices[Z: ZZ, Y: YY, X: XX, C: CC])
        tile = np.moveaxis(tile, (0, 1, 2, 3), (1, 3, 2, 0))

        return tile.astype('float')

    def label_tile(self, image_nr, pos_zxy, size_zxy, label_value):
        Z, X, Y = pos_zxy
        ZZ, XX, YY = np.array(pos_zxy) + size_zxy
        C, original_label_value = self._mapped_label_value_to_original(
            label_value)

        slices = self._open_label_file(image_nr)
        if slices is None:
            # return tile with False values
            return np.zeros(size_zxy) != 0

        tile = np.array(slices[Z: ZZ, Y: YY, X: XX, C])
        tile = np.moveaxis(tile, (0, 1, 2), (0, 2, 1))
        tile = (tile == original_label_value)
        return tile

    @lru_cache(maxsize=10)
    def _open_label_file(self, image_nr):
        # memmap is slow, so we must cache it to be fast!
        path = self.img_path / self.filenames[image_nr].img
        label_filename = self.filenames[image_nr].lbl

        if label_filename is None:
            logger.warning(
                'no label matrix file found for image file %s', str(image_nr))
            return None

        path = self.label_path / label_filename
        logger.debug('Trying to load labelmat %s', path)

        lbl_data = memmap(path)

        return self.fix_dims(lbl_data, path)  # shape order: z, y, x, c

    @staticmethod
    def calc_label_values_mapping(original_labels):
        '''
        Assign unique labelvalues to original labelvalues.

        For multichannel label images it might happen, that identical
        labels occur in different channels.
        to avoid conflicts, original labelvalues are mapped to unique values
        in ascending order 1, 2, 3, 4...
        This is defined in self.labelvalue_mapping:

        [{orig_label1: 1, orig_label2: 2}, {orig_label1: 3, orig_label2: 4},..]

        Each element of the list correponds to one label channel.
        Keys are the original labels, values are the assigned labels that
        will be seen by the Dataset object.

        Parameters
        ----------
        original_labels : array_like
            List of original label values.

        Returns
        -------
        dict
            Labelvalue mapping with original labels as key and new label as
            value.
        '''
        new_labels = itertools.count(1)

        label_mappings = [
            {l: next(new_labels) for l in sorted(labels_per_channel)}
            for labels_per_channel in original_labels
        ]

        logger.debug('Label values are mapped to ascending values:')
        logger.debug(label_mappings)
        return label_mappings

    @lru_cache(maxsize=1)
    def original_label_values_for_all_images(self):
        '''
        Get all unique label values per image.

        Returns
        -------
        list
            List of sets. Each set corresponds to 1 label channel.
            each set contains the label values of that channel.
            E.g. `[{91, 109, 150}, {90, 100}]` for two label channels
        '''
        labels_per_channel = []

        for image_nr in range(self.image_count()):
            slices = self._open_label_file(image_nr)
            if slices is None:
                continue

            C = slices.shape[-1]
            labels = [np.unique(slices[..., c]) for c in range(C)]

            labels = [set(labels) - {0} for labels in labels]

            labels_per_channel = [l1.union(l2)
                                  for l1, l2 in zip_longest(labels_per_channel,
                                                            labels,
                                                            fillvalue=set())]

        return labels_per_channel

    @lru_cache(maxsize=1500)
    def label_count_for_image(self, image_nr):
        '''
        Get number of labels per labelvalue for an image.

        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        dict
        '''
        slices = self._open_label_file(image_nr)
        if slices is None:
            return None

        C = slices.shape[-1]
        labels = [np.unique(slices[..., c]) for c in range(C)]

        original_label_count = [{l: np.count_nonzero(slices[..., c] == l)
                                 for l in labels[c] if l > 0}
                                for c in range(C)]

        label_count = {self.labelvalue_mapping[c][l]: count
                       for c, orig in enumerate(original_label_count)
                       for l, count in orig.items()}
        return label_count
