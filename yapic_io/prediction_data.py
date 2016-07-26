import yapic_io.utils as ut

class Prediction_data(object):
    '''


    
    '''
    

    def __init__(self, dataset, size_zxy, padding_zxy=(0,0,0)):
        self.dataset = dataset
        self.size_zxy = size_zxy
        self.tpl_pos = self._compute_pos_zxy()
        
        
    def _compute_pos_zxy(self):
        tpl_pos = []
        for img_nr in list(range(self.dataset.n_images)):    
            img_shape_czxy = self.dataset.get_img_dimensions(img_nr)
            print(self.dataset.get_img_dimensions(img_nr))
            img_shape_zxy = img_shape_czxy[1:]
            print(img_shape_zxy)
            tpl_pos =tpl_pos + [(img_nr, pos) for pos in ut.compute_pos(img_shape_zxy, self.size_zxy)]

        return tpl_pos    
    

    

