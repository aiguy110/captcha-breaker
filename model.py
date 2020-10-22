import cv2
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
        # self.backbone_layers.append( layers.Conv2D(64, (3,3), activation='relu', padding='same') )
        # self.backbone_layers.append( layers.Conv2D(128, (3,3), activation='relu', padding='same') )
        # self.backbone_layers.append( layers.Conv2D(128, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.MaxPool2D((2,2)) )
        # self.backbone_layers.append( layers.Conv2D(128, (3,3), activation='relu', padding='same') )
        # self.backbone_layers.append( layers.Conv2D(256, (3,3), activation='relu', padding='same') )
        # self.backbone_layers.append( layers.Conv2D(256, (3,3), activation='relu', padding='same') )
        self.backbone_layers.append( layers.MaxPool2D((2,2)) )
        # self.backbone_layers.append( layers.Conv2D(256, (3,3), activation='relu', padding='same') )
        # self.backbone_layers.append( layers.Conv2D(512, (3,3), padding='same') ) 
        # self.backbone_layers.append( layers.Conv2D(512, (3,3), padding='same') ) 
        self.backbone_layers.append( layers.MaxPool2D((2,2)) )
        # self.backbone_layers.append( layers.Conv2D(512, (3,3), activation='relu', padding='same') )
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
            box_offset_factors     = box_offset_slice#activations.sigmoid( box_offset_slice )
            box_scale_factors      = box_scale_slice#activations.exponential( box_scale_slice )

            return layers.Concatenate(axis=-1)( [class_activations, objectness_activations, box_offset_factors, box_scale_factors] )
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
        scale_pre_activations = np.log( np.array([rect_width, rect_height]) / scale_prior )

        normalized_offset = ( CaptchaBreaker.rect_center( rect ) - cell_offset * cell_size ) / cell_size
        offset_pre_activations = np.log( (1-normalized_offset) / normalized_offset )

        return np.concatenate( (offset_pre_activations, scale_pre_activations) )

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
    
    @staticmethod
    def yolo_loss(Y_true, Y_pred):
        #print(Y_true.shape, Y_pred.shape)
        classifier_activations_true = Y_true[:, :, :, :62]
        classifier_logits_pred      = Y_pred[:, :, :, :62]
        classifier_loss_pre_wieghting = tf.nn.softmax_cross_entropy_with_logits(classifier_activations_true, classifier_logits_pred)
        classifier_loss_final = tf.math.reduce_sum( tf.math.multiply(classifier_loss_pre_wieghting, Y_true[:, :, :, 67]), axis=[1,2] )

        objectness_activations_true = Y_true[:, :, :, 62]
        objectness_logits_pred      = Y_pred[:, :, :, 62]
        objectness_loss_pre_wieghting = tf.nn.sigmoid_cross_entropy_with_logits(objectness_activations_true, objectness_logits_pred)
        objectness_loss_final = tf.math.reduce_sum( tf.math.multiply(objectness_loss_pre_wieghting, Y_true[:, :, :, 68]), axis=[1,2] )

        bounding_box_pre_activations_true = Y_true[:, :, :, 63:67]
        bounding_box_pre_activations_pred = Y_pred[:, :, :, 63:67]
        bounding_box_sum_square_diffs = tf.math.reduce_sum( tf.math.square( bounding_box_pre_activations_true - bounding_box_pre_activations_pred ), axis=-1 )
        bounding_box_loss_final = tf.math.reduce_sum( tf.math.multiply(bounding_box_sum_square_diffs, Y_true[:, :, :, 69]), axis=[1,2] )

        return classifier_loss_final + objectness_loss_final + bounding_box_loss_final
    
    @staticmethod
    def load_and_preprocess_image(image_path):
        im = cv2.imread(image_path)
        X = np.zeros(im.shape, dtype='float32')
        for c in range(3):
            X[:,:,c] = im[:,:,c] / np.amax(im[:,:,c])
        
        return X 