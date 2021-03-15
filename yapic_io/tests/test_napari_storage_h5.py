import napari_storage_h5 as h5na
import matplotlib.pyplot as plt
import numpy as np


file_path = '../Data/connector_test.h5'
im_path = '../data/single_images/leaves_1.tif'
napari_project = h5na.NapariStorage(file_path, 3)
leaves_1_data = plt.imread(im_path)

class TestNapariStorage():
    def test_get_array_data(self):
        test_data = napari_project.get_array_data('image', 'leaves_1')
        assert np.array_equal(leaves_1_data, test_data)
    
    def test_excluded_layers(self):
        test_dict = napari_project.excluded_layers()
        out_dict = {'image': {'leaves_stack'},
                    'points': {'leaves_1_points'}}
        assert out_dict == test_dict

    def test_n_dims(self):
        test_n_1 = napari_project.n_dims('image', 'leaves_stack')
        test_n_2 = napari_project.n_dims('labels', 'leaves_stack_label')
        assert test_n_1 == 4
        assert test_n_2 == 3

    def test_dim_check(self):
        assert napari_project.dim_check('image', 'leaves_1')
        assert not napari_project.dim_check('image', 'leaves_stack')

    def test_get_labels_names(self):
        out = set(['leaves_1_label', 'leaves_stack_label'])
        assert set(napari_project.get_labels_names()) == out

    def test_get_image_names(self):
        assert napari_project.get_image_names() == ['leaves_1']

    def test___len__(self):
        assert len(napari_project) == 3

    def test_number_of_labels(self):
        assert napari_project.number_of_labels() == 2

    def test_number_of_images(self):
        assert napari_project.number_of_images() == 1