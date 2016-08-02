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
    

    

