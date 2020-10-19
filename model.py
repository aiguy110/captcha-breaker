import tensorflow as tf
from tensorflow.keras import activations, layers

class CaptchaBreaker(tf.keras.Model):
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
        self.backbone_layers.append( layers.Conv2D(67, (3,3) ) )

    def call(self, inputs, training=False):
        x = inputs
        for layer in self.backbone_layers:
            x = layer(x)
        
        class_slice      = x[:,:,:,  :62] # 26*2 letters + 10 digits
        objectness_slice = x[:,:,:,62:63] # objectness
        box_offset_slice = x[:,:,:,63:65] # (x, y) detection box offset factors from center of cell 
        box_scale_slice  = x[:,:,:,65:67] # (w, h) detection box scale factors

        class_logits      = activations.sigmoid( class_slice )
        objectness_logits = activations.sigmoid( objectness_slice )
        box_offset_factor = activations.sigmoid( box_offset_slice )
        box_scale_factors = activations.exponential( box_scale_slice )

        return layers.Concatenate(axis=-1)( [class_logits, objectness_logits, box_offset_factor, box_scale_factors] )

