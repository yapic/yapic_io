import yapic_io.utils as ut

class Prediction_data(object):
    '''


    
    '''
    

    def __init__(self, dataset, size_zxy, padding_zxy=(0,0,0)):
        self.dataset = dataset
        

        self.size_zxy = size_zxy
        self.padding_zxy = padding_zxy
        self.channels = self.dataset.get_channels()
        self.labels = self.dataset.get_label_values()
        
        self.pos_zxy = None
        self.image_nr = None

        self._tpl_pos_all = self._compute_pos_zxy()


        



        
        
        
    def __len__(self):
        return len(self._tpl_pos_all)


    def __getitem__(self, position):

        image_nr, pos_zxy = self._tpl_pos_all[position]
        
        self.image_nr = image_nr
        self.pos_zxy = pos_zxy
        
        return self
       

    def put_probmap_data(self, probmap_data):
        if len(probmap_data.shape) != 4: 
            raise ValueError(\
                '''no valid dimension for probmap template: 
                   shape is %s, len of shape should be 4: (c,z,x,y)'''\
                                % str(probmap_data.shape))

        n_c, n_z, n_x, n_y = probmap_data.shape

        if n_c != len(self.labels):
            raise ValueError(\
                '''template must have %s channels, one channel for each
                   label in follwoing label order: %s'''\
                                % (str(len(self.labels)), str(self.labels)))

        if (n_z, n_x, n_y) != self.size_zxy:
            raise ValueError(\
                '''zxy shape of probmap template is not valid: 
                   is %s, should be %s''' \
                   % ((str((n_z, n_x, n_y)), str(self.size_zxy))))    
          
    def put_probmap_data_for_label(self, probmap_data, label):
        if len(probmap_data.shape) != 3: 
            raise ValueError(\
                '''no valid dimension for probmap template: 
                   shape is %s, len of shape should be 3: (z,x,y)'''\
                                % str(probmap_data.shape))

        n_z, n_x, n_y = probmap_data.shape
        if (n_z, n_x, n_y) != self.size_zxy:
            raise ValueError(\
                '''zxy shape of probmap template is not valid: 
                   is %s, should be %s''' \
                   % ((str((n_z, n_x, n_y)), str(self.size_zxy))))    

            
        if label not in self.labels:
            raise ValueError('label %s not found in labels %s' % (str(label), str(self.labels)))

        self.dataset.put_prediction_template(probmap_data, self.pos_zxy, self.image_nr, label)    
            
    def get_pixels(self):
        return self.dataset.get_multichannel_pixel_template(\
            self.image_nr, self.pos_zxy, self.size_zxy, self.channels,\
            pixel_padding=self.padding_zxy)   
        
    def _compute_pos_zxy(self):
        tpl_pos = []
        for img_nr in list(range(self.dataset.n_images)):    
            img_shape_czxy = self.dataset.get_img_dimensions(img_nr)
            print(self.dataset.get_img_dimensions(img_nr))
            img_shape_zxy = img_shape_czxy[1:]
            print(img_shape_zxy)
            tpl_pos =tpl_pos + [(img_nr, pos) for pos in ut.compute_pos(img_shape_zxy, self.size_zxy)]

        return tpl_pos    
    

    

