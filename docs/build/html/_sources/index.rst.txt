Welcome to yapic_io!
====================

*yapic_io* provides classes for dynamic data-binding of arbitrary large bio image
data and associated label data. *yapic_io* is designed as an input-output layer
for training of fully convolutional neural networks, as well as writing predicted
data to tiff image files.

* Supported pixel data sources are tiff and bigtiff files.
* Supported label data sources are tiff and bigtiff files, as well as Ilastik
  Project Files (ilp).

This library provides two types of :class:`yapic_io.minibatch.Minibatch`:
:class:`yapic_io.training_batch.TrainingBatch` and :class:`yapic_io.prediction_batch.PredictionBatch`.

Each minibatch has a :class:`yapic_io.dataset.Dataset`, which in turn have a
:class:`yapic_io.connector.Connector`.


Contents:

.. toctree::
   :maxdepth: 2

   installation
   getting_started
   api



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
