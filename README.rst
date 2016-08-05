Documentation
=============


yapic_io provides flexible data binding to image collections of arbitrary size.


Its aim is to provide a convenient image data interface for training of
fully convolutional neural networks, as well as automatic handling of 
prediction data output for a trained classifier.

Following problems occuring with training/classification are handeled by yapic_io:

- Images of different sizes in z,x, and y can be applied to the same convolutional   
  network. This is implemented by splitting images into smaller templates of identical sizes on the fly. The size these templates corresponds to the size of
  the convolutional network's input layer. 

- Due to lazy data loading, images can be extremely large.

- Image dimensions can be up to 4D (multchannel z-stack), as e.g. required for   
  bioimages.

- Data augmentation for classifier training in built in.  






Code example
============

Classifier training:

    >>> from yapic_io.factories import make_tiff_interface
    >>>
    >>> #define data loacations
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = 'yapic_io/test_data/tmp/'
    >>> 
    >>> tpl_size = (1,5,4) # size of network output layer in zxy
    >>> padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
    >>>
    >>> # make minibatch mb and prediction interface p with TiffConnector binding
    >>> m, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size, padding_zxy=padding) 
    >>>
    >>> counter=0
    >>> for mini in m:
    ...     weights = mini.weights
    ...     #shape of weights is (6,3,1,5,4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
    ...        
    ...     pixels = mini.pixels 
    ...     # shape of pixels is (3,1,9,8) : 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
    ...     #here: apply training on mini.pixels and mini.weights
    ...     counter += 1
    ...     if counter > 10: #m is infinite
    ...         break

Prediction:

    >>> from yapic_io.factories import make_tiff_interface
    >>>
    >>> #define data loacations
    >>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
    >>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
    >>> savepath = 'yapic_io/test_data/tmp/'
    >>> 
    >>> tpl_size = (1,5,4) # size of network output layer in zxy
    >>> padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
    >>>
    >>> # make minibatch mb and prediction interface p with TiffConnector binding
    >>> _, p = make_tiff_interface(pixel_image_dir, label_image_dir, savepath, tpl_size, padding_zxy=padding) 
    >>> len(p) #give total the number of templates that cover the whole bound tiff files 
    510
    >>>
    >>> #classify the whole bound dataset
    >>> counter = 0 #needed for mock data
    >>> for item in p:
    ...     pixels_for_classifier = item.get_pixels() #input for classifier
    ...     mock_classifier_result = np.ones(tpl_size) * counter #classifier output
    ...     #pass classifier results for each class to data source
    ...     item.put_probmap_data_for_label(mock_classifier_result, label=91)
    ...     item.put_probmap_data_for_label(mock_classifier_result, label=109)
    ...     item.put_probmap_data_for_label(mock_classifier_result, label=150)
    ...     counter += 1
    >>>





