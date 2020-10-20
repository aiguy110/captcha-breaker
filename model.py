import string
import numpy as np
import tensorflow as tf
from tensorflow.keras import activations, layers

class CaptchaBreaker(tf.keras.Model):
    scale_prior = 16
    alphanum = string.ascii_lowercase + string.ascii_uppercase + string.digits

    def __init__(self):
        super(CaptchaBreaker, self).__init__()
        self.backbone_layers = [] 
        self.backbone_layers.append( layers.Conv2D(32, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.BatchNormalization() )
        self.backbone_layers.append( layers.Conv2D(64, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.BatchNormalization() )
        self.backbone_layers.append( layers.MaxPool2D((2,2)) )
        self.backbone_layers.append( layers.Conv2D(64, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.BatchNormalization() )
        self.backbone_layers.append( layers.Conv2D(128, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.BatchNormalization() )
        self.backbone_layers.append( layers.MaxPool2D((2,2)) )
        self.backbone_layers.append( layers.Conv2D(128, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.BatchNormalization() )
        self.backbone_layers.append( layers.Conv2D(67, (3,3), padding='same') ) 

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.backbone_layers:
            x = layer(x)
        
        if not training:
            class_slice      = x[:,:,:,  :62] # 26*2 letters + 10 digits
            objectness_slice = x[:,:,:,62:63] # objectness
            box_offset_slice = x[:,:,:,63:65] # (x, y) detection box offset factors from center of cell 
            box_scale_slice  = x[:,:,:,65:67] # (w, h) detection box scale factors

            class_activations      = activations.softmax( class_slice )
            objectness_activations = activations.sigmoid( objectness_slice )
            box_offset_factors     = activations.sigmoid( box_offset_slice )
            box_scale_factors      = activations.exponential( box_scale_slice )

            return layers.Concatenate(axis=-1)( [class_logits, objectness_logits, box_offset_factor, box_scale_factors] )
        else:
            return x
        

    @staticmethod
    def decode_rect(arr, cell_offset, cell_size, scale_prior=None):
        if scale_prior == None:
            scale_prior = CaptchaBreaker.scale_prior

        init_offset = cell_offset[:2] * cell_size
        
        def sigma(x):
            return 1 / ( 1 + np.exp(-x) )
        
        output_offsets = sigma(arr[:2]) * cell_size
        full_offset = init_offset + output_offsets 
        
        w, h = np.exp(arr[2:]) * scale_prior
        return { 'left': full_offset[0] - w/2, 'right': full_offset[0] + w/2, 'top': full_offset[1] - h/2, 'bottom': full_offset[1] + h/2 }
    
    @staticmethod
    def encode_rect(rect, cell_offset, cell_size, scale_prior=None):
        if scale_prior == None:
            scale_prior = CaptchaBreaker.scale_prior
        
        rect_width  = rect['right' ] - rect['left']
        rect_height = rect['bottom'] - rect['top' ]
        scale_activations = np.log( np.array([rect_width, rect_height]) / scale_prior )

        normalized_offset = ( CaptchaBreaker.rect_center( rect ) - cell_offset * cell_size ) / cell_size
        offset_activations = np.log( (1-normalized_offset) / normalized_offset )

        return np.concatenate( (offset_activations, scale_activations) )

    @staticmethod
    def rect_center(rect):
        return np.array( [(rect['left']+rect['right'])/2, (rect['top']+rect['bottom'])/2] )

    @staticmethod
    def point_is_inside_rect(point, rect):
        x, y = point
        if rect['left'] < x < rect['right'] and rect['top'] < y < rect['bottom']:
            return True
        else:
            return False