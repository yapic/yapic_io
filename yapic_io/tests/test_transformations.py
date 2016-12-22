from unittest import TestCase


import numpy as np
import yapic_io.transformations as tf

class TestTransformations(TestCase):
    
    
    def test_mirror_edges_dim0(self):
        #input data
        m = [[1,1,1]\
            ,[2,2,2]\
            ,[3,3,3]]   
        m = np.array(m)

        #validation data
        val = [[3,3,3]\
              ,[2,2,2]\
              ,[1,1,1]\
              ,[2,2,2]\
              ,[3,3,3]\
              ,[2,2,2]\
              ,[1,1,1]] 
        val = np.array(val)   

        n_pix=(2,0)

        #output data
        out = tf.mirror_edges(m,n_pix) 


        
        self.assertTrue(np.alltrue(out == val))


    def test_get_transform(self):
        image = np.zeros((4,5,6)) #3 dim image
        rotation_angle=45
        shear_angle=45

        # with self.assertRaises(ValueError) as context:
        #     tf.get_transform(image, rotation_angle, shear_angle)
        # print(context.exception)
        self.assertRaises(ValueError, lambda: tf.get_transform(image, rotation_angle, shear_angle))  
        #self.assertTrue('image has 3 dimensions instead of 2' in context.exception)

    def test_warp_image_2d_value_error(self):
        image = np.zeros((4,5,6)) #3 dim image
        rotation_angle=45
        shear_angle=45

        # with self.assertRaises(ValueError) as context:
        #     tf.warp_image_2d(image, rotation_angle, shear_angle)
        # print('context exception:')
        # print(type(context.exception))
        # print(context.exception)
        #print('hallo')
        self.assertRaises(ValueError, lambda: tf.warp_image_2d(image, rotation_angle, shear_angle))  
        #self.assertTrue('''image has 3 dimensions instead of 2''' == context.exception)  
        

    def test_warp_image_2d(self):
        '''
        test 45 degrees center rotation with 3x3 matrix
        '''
        rotation_angle=45
        shear_angle=0

        im = [[0,1,0]\
             ,[0,1,0]\
             ,[0,1,0]]    
        
        validation =  [[0,0,1]\
                     ,[0,1,0]\
                     ,[1,0,0]]  
        
        im = np.array(im)
        validation = np.array(validation)
        
        rot = tf.warp_image_2d(im, rotation_angle, shear_angle)
        print(validation)
        print(rot)
        self.assertTrue((rot==validation).all())

    
    def test_warp_image_2d_stack(self):
        rotation_angle=45
        shear_angle=0

        im = [[[0,1,0]\
              ,[0,1,0]\
              ,[0,1,0]],\
              [[0,2,0]\
              ,[0,2,0]\
              ,[0,2,0]]]

        val = [[[0,0,1]\
              ,[0,1,0]\
              ,[1,0,0]],\
              [[0,0,2]\
              ,[0,2,0]\
              ,[2,0,0]]]      
        val = np.array(val)      
        im = np.array(im)      
        rot = tf.warp_image_2d_stack(im, rotation_angle, shear_angle)
        print(rot)
        self.assertTrue((rot==val).all())
        self.assertEqual(len(rot.shape), 3)


    def test_warp_image_2d_stack_4d(self):
        rotation_angle=45
        shear_angle=0

        im = [[[[0,1,0]\
              ,[0,1,0]\
              ,[0,1,0]],\
              [[0,2,0]\
              ,[0,2,0]\
              ,[0,2,0]]],\
              [[[0,1,0]\
              ,[0,1,0]\
              ,[0,1,0]],\
              [[0,2,0]\
              ,[0,2,0]\
              ,[0,2,0]]]]

        val = [[[[0,0,1]\
              ,[0,1,0]\
              ,[1,0,0]],\
              [[0,0,2]\
              ,[0,2,0]\
              ,[2,0,0]]],\
              [[[0,0,1]\
              ,[0,1,0]\
              ,[1,0,0]],\
              [[0,0,2]\
              ,[0,2,0]\
              ,[2,0,0]]]]



        val = np.array(val)      
        im = np.array(im)      
        rot = tf.warp_image_2d_stack(im, rotation_angle, shear_angle)
        print(rot)
        np.testing.assert_array_equal(rot, val)
        self.assertEqual(len(rot.shape), 4)    


    # def test_calc_warping_shift(self):

    #     img_shape = (7,7)

    #     tf.calc_warping_shift(img_shape, 0, 20)

    #     self.assertTrue(False)



    # def test_warp_image_2d_90(self):
    #     '''
    #     test 45 degrees center rotation with 3x3 matrix
    #     '''
    #     rotation_angle=90
    #     shear_angle=0

    #     im = [[0,1,0]\
    #          ,[0,1,0]\
    #          ,[0,1,0]]    
        
    #     validation =  [[0,0,0]\
    #                  ,[1,1,1]\
    #                  ,[0,0,0]]  
        
    #     im = np.array(im)
    #     validation = np.array(validation)
        
    #     rot = tf.warp_image_2d(im, rotation_angle, shear_angle)
    #     print(rot)
    #     self.assertTrue((rot==validation).all())    
        

      

