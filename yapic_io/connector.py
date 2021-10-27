from abc import ABCMeta, abstractmethod
import logging
import os
logger = logging.getLogger(os.path.basename(__file__))


def io_connector(image_path, label_path, *args, **kwds):
    '''
    Returns either a TiffConnector or an NapariConnector, depending on
    input files.

    Parameters
    ----------
    image_path : str or list of str
        Either wildcard or list of paths to pixel data in tiff format.
    label_path : str of list of str
        Either wildcard or list of paths to pixel data in tiff format
        returns a TiffConnector. If a path to a single Ilastik ilp
        file is given, an NapariConnector is returned.

    Returns
    -------
    Connector
        Either NapariConnector or TiffConnector

    See Also
    --------
    yapic_io.tiff_connector.TiffConnector
    yapic_io.napari_connector.NapariConnector
    '''
    from yapic_io.tiff_connector import TiffConnector
    from yapic_io.napari_connector import NapariConnector

    if label_path.endswith('.h5'):
        logger.info('Napari project file detected')
        return NapariConnector(image_path, label_path)
    else:
        logger.info('Tiff files detected.')
        return TiffConnector(image_path, label_path, *args, **kwds)


class Connector(metaclass=ABCMeta):
    '''
    Interface to pixel and label data source for classifier
    training and prediction.

    - Connects to a fixed collection of images and corresponding labels
    - Provides methods for getting sizes of all images
    - Provides methods for getting image subsets and label coordinates.


    Prerequisites for the Image data:

    - Images are 4D datasets with following dimensions: (channel, z, x, y)
    - All images must have the same nr of channels but can vary in z, x, and y.
    - Time series are not supported. However, time frames can be modeled as
      individual images.
    - Images with lower dimensions can be modeled. E.g. grayscale single plane
      images
      have the shape (1, 1, width, height).

    The Connector methods are used by the Dataset class.
    '''

    def __init__(self):
        '''
        In polling mode, the dataset will repeatedly fetch
        tiles for training (using get_tile) until labels of a
        certain value are present.
        This is efficient in following situation:
        - relatively small images
        - dense labelling

        For very large images with sparse labelling,
        the by_label_index mode is recommended (implemented in
        Coordinate_connector class)
        '''


    @abstractmethod
    def image_count(self):
        '''
        Get number of images.

        Returns
        -------
        int
            Number of images in the dataset.
        '''
        pass

    @abstractmethod
    def label_count_for_image(self, image_nr):
        '''
        Returns for each label value the number of labels for this image

        Parameters
        ----------
        image_nr : int
            Index of image.

        Returns
        -------
        dict
            Counts per label and per image.
        '''
        pass

    @abstractmethod
    def get_tile(self, image_nr=None, pos=None, size=None):
        '''
        Get 4D subsection of an image.

        Parameters
        ----------
        image_nr : int
            Index of image.
        pos : (channel, zslice, x, y)
            Upper left position of subsection

        Returns
        -------
        numpy.ndarray
            4D subsection of image as numpy array
        '''
        pass

    @abstractmethod
    def label_tile(self, image_nr, pos_zxy, size_zxy, label_value):
        '''
        Get 3d zxy boolean matrix where positions of the requested label
        are indicated with True. Only mapped labelvalues can be requested.

        dimension order: (z, x, y)

        Parameters
        ----------
        image_nr : int
            Index of image.
        pos_zxy : (zslice, x, y)
            Upper left position of subsection.
        size_zxy : (nr_zslices, nr_x, nr_y)
            Upper left position of subsection.
        label_value : int
            Id of the label.

        Returns
        -------
        numpy.ndarray
            3D subsection of labelmatrix as boolean mask in dimension order
            (z, x, y)
        '''

        pass

    @abstractmethod
    def put_tile(self, pixels, pos_zxy, image_nr, label_value):
        '''
        Puts probabilities (pixels) for a certain label to the data storage.

        These probabilities are prediction values from a classifier (i.e. the
        output of the classier)

        Parameters
        ----------
        pixels : numpy.ndarray
            3D matrix of probability values with shape (z, x, y)
        pos_zxy : (z, x, y)
            Upper left position of pixels in source image_nr.
        image_nr : int
            Index of image.
        label_value : int
            Id of the label.

        Returns
        -------
        bool
            True in case of successful write.
        '''

    @abstractmethod
    def image_dimensions(self, image_nr):
        '''
        Get dimensions of the dataset.


        Parameters
        ----------
        image_nr : int
            index of image

        Returns
        -------
        (nr_channels, nr_zslices, nr_x, nr_y)
            Image shape.
        '''
        pass
