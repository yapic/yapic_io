[![Build Status](https://travis-ci.com/yapic/yapic_io.svg?branch=master)](https://travis-ci.com/yapic/yapic_io)
[![Documentation Status](https://readthedocs.org/projects/yapic-io/badge/?version=latest)](https://yapic-io.readthedocs.io/en/latest/?badge=latest)

# yapic_io




yapic_io provides flexible data binding to image collections of arbitrary size.


Its aim is to provide a convenient image data interface for training of
fully convolutional neural networks, as well as automatic handling of
prediction data output for a trained classifier.

yapic_io is designed as a convenient image data input/output interface for
libraries such as Theano or TensorFlow.


Following problems occuring with training/classification are handeled by yapic_io:

- Images of different sizes in z,x, and y can be applied to the
  same convolutional network. This is implemented by sliding windows. The size these windows correspond to the size of the convolutional network's input layer.

- Due to lazy data loading, images can be extremely large.

- Image dimensions can be up to 4D (multchannel z-stack), as e.g. required
  for bioimages.

- Data augmentation for classifier training in built in.

- Made for sparsly labelled datasets: Training data is only (randomly) picked
  from regions where labels are present.

- Usually, input layers of CNNs are larger than output layers. Thus, pixels
  located at image edges are normally not classified. With yapic_io also
  edge pixels are classified. This is achieved by mirroring pixel data in edge
  regions. As a result, output classification images have identical dimensions as source images and can be overlayed easily.



## Example

Classifier training:

```
>>> from yapic_io import TiffConnector, Dataset, TrainingBatch
>>>
>>> #define data locations
>>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
>>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
>>> savepath = 'yapic_io/test_data/tmp/'
>>>
>>>
>>> tpl_size = (1,5,4) # size of network output layer in zxy
>>> padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
>>>
>>> c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath)
>>> train_data = TrainingBatch(Dataset(c), tpl_size, padding_zxy=padding)
>>>
>>> counter=0
>>> for mini in train_data:
...     weights = mini.weights
...     #shape of weights is (6,3,1,5,4) : batchsize 6 , 3 label-classes, 1 z, 5 x, 4 y
...
...     pixels = mini.pixels()
...     # shape of pixels is (6,3,1,9,8) : 3 channels, 1 z, 9 x, 4 y (more xy due to padding)
...
...     #here: apply training on mini.pixels and mini.weights (use theano, tensorflow...)
...     my_train_function(pixels, weights)
...
...     counter += 1
...     if counter > 10: #m is infinite
...         break
```
Prediction:
```
>>> from yapic_io import TiffConnector, Dataset, PredictionBatch
>>>
>>> #mock classification function
>>> def classify(pixels, value):
...     return np.ones(pixels.shape) * value
>>>
>>> #define data loacations
>>> pixel_image_dir = 'yapic_io/test_data/tiffconnector_1/im/*.tif'
>>> label_image_dir = 'yapic_io/test_data/tiffconnector_1/labels/*.tif'
>>> savepath = 'yapic_io/test_data/tmp/'
>>>
>>> tpl_size = (1,5,4) # size of network output layer in zxy
>>> padding = (0,2,2) # padding of network input layer in zxy, in respect to output layer
>>>
>>> c = TiffConnector(pixel_image_dir, label_image_dir, savepath=savepath)
>>> prediction_data = PredictionBatch(Dataset(c))
>>> len(prediction_data) #give the total number of templates that cover the whole bound tifffiles
510
>>>
>>> #classify the whole bound dataset
>>> counter = 0 #needed for mock data
>>> for item in prediction_data:
...     pixels_for_classifier = item.pixels() #input for classifier
...     mock_classifier_result = classify(pixels, counter) #classifier output
...
...     #pass classifier results for each class to data source
...     item.put_probmap_data(mock_classifier_result)
...
...     counter += 1 #counter for generation of mockdata
>>>
```

## Buils API docs

```
cd docs
sphinx-apidoc -o source ../yapic_io
make html
```


Developed by the CRFS (Core Research Facilities) of the DZNE (German Center
for Neurodegenerative Diseases).
