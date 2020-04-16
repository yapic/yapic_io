from unittest import TestCase
import numpy as np
import yapic_io.transformations as tf
from numpy.testing import assert_array_equal


class TestTransformations(TestCase):

    def test_get_transform(self):
        image = np.zeros((4, 5, 6))  # 3 dim image
        rotation_angle = 45
        shear_angle = 45

        self.assertRaises(ValueError,
                          lambda: tf.get_transform(image,
                                                   rotation_angle,
                                                   shear_angle))

    def test_warp_image_2d_value_error(self):
        image = np.zeros((4, 5, 6))  # 3 dim image
        rotation_angle = 45
        shear_angle = 45

        self.assertRaises(ValueError,
                          lambda: tf.warp_image_2d(image,
                                                   rotation_angle,
                                                   shear_angle))

    def test_warp_image_2d(self):
        '''
        test 45 degrees center rotation with 3x3 matrix
        '''
        rotation_angle = 45
        shear_angle = 0

        im = [[0, 1, 0],
              [0, 1, 0],
              [0, 1, 0]]

        validation = [[0, 0, 1],
                      [0, 1, 0],
                      [1, 0, 0]]

        im = np.array(im)
        validation = np.array(validation)

        rot = tf.warp_image_2d(im, rotation_angle, shear_angle)

        assert_array_equal(rot.astype(int), validation)

    def test_warp_image_2d_stack(self):

        rotation_angle = 45
        shear_angle = 0

        im = [[[0, 1, 0],
               [0, 1, 0],
               [0, 1, 0]],
              [[0, 2, 0],
               [0, 2, 0],
               [0, 2, 0]]]

        val = [[[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]],
               [[0, 0, 2],
                [0, 2, 0],
                [2, 0, 0]]]

        val = np.array(val)
        im = np.array(im)
        rot = tf.warp_image_2d_stack(im, rotation_angle, shear_angle)

        assert_array_equal(rot.astype(int), val)
        self.assertEqual(len(rot.shape), 3)

    def test_warp_image_2d_stack_4d(self):
        rotation_angle = 45
        shear_angle = 0

        im = [[[[0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]],
              [[0, 2, 0],
               [0, 2, 0],
               [0, 2, 0]]],
              [[[0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]],
               [[0, 2, 0],
                [0, 2, 0],
                [0, 2, 0]]]]

        val = [[[[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]],
                [[0, 0, 2],
                 [0, 2, 0],
                 [2, 0, 0]]],
               [[[0, 0, 1],
                 [0, 1, 0],
                 [1, 0, 0]],
                [[0, 0, 2],
                 [0, 2, 0],
                 [2, 0, 0]]]]

        val = np.array(val)
        im = np.array(im)
        rot = tf.warp_image_2d_stack(im, rotation_angle, shear_angle)
        print(rot)
        assert_array_equal(rot.astype(int), val)
        self.assertEqual(len(rot.shape), 4)

    def test_flip_image_2d_4dstack(self):

        im = np.array([[[[0., 0., 3., 0., 0.],
                         [0., 0., 3., 0., 0.],
                         [1., 1., 5., 2., 2.],
                         [0., 0., 4., 0., 0.],
                         [0., 0., 4., 0., 0.]]]])

        ud_val = np.array([[[[0., 0., 3., 0., 0.],
                             [0., 0., 3., 0., 0.],
                             [2., 2., 5., 1., 1.],
                             [0., 0., 4., 0., 0.],
                             [0., 0., 4., 0., 0.]]]])

        lr_val = np.array([[[[0., 0., 4., 0., 0.],
                             [0., 0., 4., 0., 0.],
                             [1., 1., 5., 2., 2.],
                             [0., 0., 3., 0., 0.],
                             [0., 0., 3., 0., 0.]]]])

        udlr_val = np.array([[[[0., 0., 4., 0., 0.],
                               [0., 0., 4., 0., 0.],
                               [2., 2., 5., 1., 1.],
                               [0., 0., 3., 0., 0.],
                               [0., 0., 3., 0., 0.]]]])

        rot90val = np.array([[[[0., 0., 1., 0., 0.],
                               [0., 0., 1., 0., 0.],
                               [4., 4., 5., 3., 3.],
                               [0., 0., 2., 0., 0.],
                               [0., 0., 2., 0., 0.]]]])

        ud = tf.flip_image_2d_stack(im, flipud=True)
        lr = tf.flip_image_2d_stack(im, fliplr=True)
        udlr = tf.flip_image_2d_stack(im, fliplr=True,
                                      flipud=True)
        rot90 = tf.flip_image_2d_stack(im, rot90=1)

        assert_array_equal(ud, ud_val)
        assert_array_equal(lr, lr_val)
        assert_array_equal(udlr, udlr_val)
        assert_array_equal(rot90, rot90val)

    def test_flip_image_2d_3dstack(self):

        im = np.array([[[0., 0., 3., 0., 0.],
                        [0., 0., 3., 0., 0.],
                        [1., 1., 5., 2., 2.],
                        [0., 0., 4., 0., 0.],
                        [0., 0., 4., 0., 0.]]])

        ud_val = np.array([[[0., 0., 3., 0., 0.],
                            [0., 0., 3., 0., 0.],
                            [2., 2., 5., 1., 1.],
                            [0., 0., 4., 0., 0.],
                            [0., 0., 4., 0., 0.]]])

        lr_val = np.array([[[0., 0., 4., 0., 0.],
                            [0., 0., 4., 0., 0.],
                            [1., 1., 5., 2., 2.],
                            [0., 0., 3., 0., 0.],
                            [0., 0., 3., 0., 0.]]])

        udlr_val = np.array([[[0., 0., 4., 0., 0.],
                              [0., 0., 4., 0., 0.],
                              [2., 2., 5., 1., 1.],
                              [0., 0., 3., 0., 0.],
                              [0., 0., 3., 0., 0.]]])

        rot90val = np.array([[[0., 0., 1., 0., 0.],
                              [0., 0., 1., 0., 0.],
                              [4., 4., 5., 3., 3.],
                              [0., 0., 2., 0., 0.],
                              [0., 0., 2., 0., 0.]]])

        ud = tf.flip_image_2d_stack(im, flipud=True)
        lr = tf.flip_image_2d_stack(im, fliplr=True)
        udlr = tf.flip_image_2d_stack(im, fliplr=True,
                                      flipud=True)
        rot90 = tf.flip_image_2d_stack(im, rot90=1)

        assert_array_equal(ud, ud_val)
        assert_array_equal(lr, lr_val)
        assert_array_equal(udlr, udlr_val)
        assert_array_equal(rot90, rot90val)
