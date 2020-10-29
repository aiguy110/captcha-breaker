import cv2
import string
import numpy as np
import tensorflow as tf

class ResBlock(tf.keras.Model): # [x] Conv2d-1x1-n1 -> Conv2d-3x3-n2 + x 
    def __init__(self, n1, n2):
        super(ResBlock, self).__init__(self)
        self.sub_layers = [
            tf.keras.layers.Conv2D(n1, (1,1), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Conv2D(n2, (3,3), padding='same'),
        ]
        self.final_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
    
    def call(self, input_tensor):
        x = input_tensor
        for layer in self.sub_layers:
            x = layer(x)
        return self.final_activation( tf.keras.layers.Add()([x, input_tensor]) ) 

class CaptchaBreaker(tf.keras.Model):
    scale_prior = 24
    label_ind_lookup = string.ascii_lowercase + string.ascii_uppercase + string.digits

    def __init__(self):
        super(CaptchaBreaker, self).__init__()
        self.backbone_layers = [] 
        self.backbone_layers.append( tf.keras.layers.Conv2D(64*3//2, (5,5), padding='same') )
        self.backbone_layers.append( tf.keras.layers.LeakyReLU(alpha=0.1) )
        
        for i in range(5):
            self.backbone_layers.append( ResBlock(32*3//2, 64*3//2) )
        self.backbone_layers.append( tf.keras.layers.Conv2D(128*3//2, (3,3), padding='same') )
        self.backbone_layers.append( tf.keras.layers.LeakyReLU(alpha=0.1) )
        self.backbone_layers.append( tf.keras.layers.MaxPool2D((2,2)) )
        
        for i in range(5):
            self.backbone_layers.append( ResBlock(64*3//2, 128*3//2) )
        self.backbone_layers.append( tf.keras.layers.Conv2D(256*3//2, (3,3), padding='same') )
        self.backbone_layers.append( tf.keras.layers.LeakyReLU(alpha=0.1) )
        self.backbone_layers.append( tf.keras.layers.MaxPool2D((2,2)) )
        
        for i in range(10):
            self.backbone_layers.append( ResBlock(128*3//2, 256*3//2) )
        self.backbone_layers.append( tf.keras.layers.Conv2D(1024, (3,3), padding='same') )
        self.backbone_layers.append( tf.keras.layers.LeakyReLU(alpha=0.1) )
        self.backbone_layers.append( tf.keras.layers.MaxPool2D((2,2)) )

        self.backbone_layers.append( tf.keras.layers.Conv2D(512, (1,1), padding='same') )
        self.backbone_layers.append( tf.keras.layers.LeakyReLU(alpha=0.1) )
        self.backbone_layers.append( tf.keras.layers.Conv2D(67, (3,3), padding='same') )

    def call(self, inputs, training=False):
        # Build model, and print memory usage info
        model_bytes = 0
        per_sample_bytes = 0
        x = inputs
        for layer in self.backbone_layers:
            x = layer(x)
            # Compute per-sample memory requirements to store this layers outputs
            elements = 1
            for dim in x.shape[1:]:
                elements *= dim
            per_sample_bytes += elements * x.dtype.size

            # Compute parameter memory requirements of this layer
            for tf_var in layer.weights:
                elements = 1
                for dim in tf_var.shape:
                    elements *= dim
                model_bytes += elements * tf_var.dtype.size
        if training:
            print(f'Model parameters require {model_bytes} bytes of memory. An additional {per_sample_bytes} bytes is required')
            print(f'for each sample in a batch (not including the input and target tensors themselves).')
        
        if False:#not training:
            class_slice      = x[:,:,:,  :62] # 26*2 letters + 10 digits
            objectness_slice = x[:,:,:,62:63] # objectness
            box_offset_slice = x[:,:,:,63:65] # (x, y) detection box offset factors from center of cell 
            box_scale_slice  = x[:,:,:,65:67] # (w, h) detection box scale factors

            class_activations      = tf.keras.activations.softmax( class_slice )
            objectness_activations = tf.keras.activations.sigmoid( objectness_slice )
            box_offset_factors     = box_offset_slice#activations.sigmoid( box_offset_slice )
            box_scale_factors      = box_scale_slice#activations.exponential( box_scale_slice )

            return tf.keras.layers.Concatenate(axis=-1)( [class_activations, objectness_activations, box_offset_factors, box_scale_factors] )
        else:
            return x
        

    @staticmethod
    def decode_rect(arr, cell_offset, cell_size, scale_prior=None):
        if scale_prior == None:
            scale_prior = CaptchaBreaker.scale_prior

        cell_center_offset = (cell_offset[:2] + np.array([0.5, 0.5])) * cell_size
        half_prior_offset = np.array([0.5, 0.5]) * scale_prior
        
        def sigma(x):
            return 1 / ( 1 + np.exp(-x) )
        
        output_offsets = sigma(arr[:2]) * scale_prior
        full_offset = cell_center_offset - half_prior_offset + output_offsets 
        
        w, h = np.exp(arr[2:]) * scale_prior
        return { 'left': full_offset[0] - w/2, 'right': full_offset[0] + w/2, 'top': full_offset[1] - h/2, 'bottom': full_offset[1] + h/2 }
    
    @staticmethod
    def encode_rect(rect, cell_offset, cell_size, scale_prior=None):
        if scale_prior == None:
            scale_prior = CaptchaBreaker.scale_prior
        
        rect_width  = rect['right' ] - rect['left']
        rect_height = rect['bottom'] - rect['top' ]
        scale_pre_activations = np.log( np.array([rect_width, rect_height]) / scale_prior )

        prior_upper_left_offset = (cell_offset[:2] + np.array([0.5, 0.5]))*cell_size - np.array([0.5, 0.5]) * scale_prior
        normalized_offset = ( CaptchaBreaker.rect_center( rect ) - prior_upper_left_offset) / scale_prior
        
        epsilon = 0.001
        normalized_offset = np.maximum( normalized_offset, [  epsilon,   epsilon] )
        normalized_offset = np.minimum( normalized_offset, [1-epsilon, 1-epsilon] )

        offset_pre_activations = - np.log( (1-normalized_offset) / normalized_offset )

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
    def rect_area(rect):
        return max(rect['right'] - rect['left'], 0) * max(rect['bottom'] - rect['top'], 0)

    @staticmethod
    def intersect_rect(rect1, rect2):
        # Note that this will always return a rect even if the input rects do not intersect.
        # In that case, the rect may be inverted left-to-right or top-to-bottom. rect_area()
        # knows to handle this situation by returning zero.
        return {
            'left'  : max( rect1['left']  , rect2['left']   ),
            'right' : min( rect1['right'] , rect2['right']  ),
            'top'   : max( rect1['top']   , rect2['top']    ),
            'bottom': min( rect1['bottom'], rect2['bottom'] )
        }

    @staticmethod
    def intersection_over_union(rect1, rect2):        
        area1 = CaptchaBreaker.rect_area( rect1 )
        area2 = CaptchaBreaker.rect_area( rect2 )
        areax = CaptchaBreaker.rect_area( CaptchaBreaker.intersect_rect(rect1, rect2) )

        return areax / (area1 + area2 - areax)
    
    @staticmethod
    def overlap_fraction(src_rect, dst_rect):
        src_area = CaptchaBreaker.rect_area( src_rect )
        int_area = CaptchaBreaker.rect_area( CaptchaBreaker.intersect_rect(src_rect, dst_rect) )
        
        return int_area / src_area

    @staticmethod
    def yolo_loss(Y_true, Y_pred):
        tf.debugging.check_numerics(Y_true, "Label tensor contained non-numerics")
        tf.debugging.check_numerics(Y_pred, "Prediction tensor contained non-numerics")

        classifier_activations_true = Y_true[:, :, :, :62]
        classifier_logits_pred      = Y_pred[:, :, :, :62]
        classifier_loss_pre_wieghting = tf.nn.softmax_cross_entropy_with_logits(classifier_activations_true, classifier_logits_pred)
        classifier_loss_final = tf.math.reduce_sum( tf.math.multiply(classifier_loss_pre_wieghting, Y_true[:, :, :, 67]), axis=[1,2] )
        tf.debugging.check_numerics(classifier_loss_pre_wieghting, "Class softmax cross entropy output contained non-numerics")

        objectness_activations_true = Y_true[:, :, :, 62]
        objectness_logits_pred      = Y_pred[:, :, :, 62]
        objectness_loss_pre_wieghting = tf.nn.sigmoid_cross_entropy_with_logits(objectness_activations_true, objectness_logits_pred)
        objectness_loss_final = tf.math.reduce_sum( tf.math.multiply(objectness_loss_pre_wieghting, Y_true[:, :, :, 68]), axis=[1,2] )
        tf.debugging.check_numerics(objectness_loss_pre_wieghting, "Objectness sigmoid cross entropy output contained non-numerics")

        bounding_box_pre_activations_true = Y_true[:, :, :, 63:67]
        bounding_box_pre_activations_pred = Y_pred[:, :, :, 63:67]
        bounding_box_sum_square_diffs = tf.math.reduce_sum( tf.math.square( bounding_box_pre_activations_true - bounding_box_pre_activations_pred ), axis=-1 )
        bounding_box_loss_final = tf.math.reduce_sum( tf.math.multiply(bounding_box_sum_square_diffs, Y_true[:, :, :, 69]), axis=[1,2] )
        tf.debugging.check_numerics(bounding_box_sum_square_diffs, "Bounding box sum of square differences output contained non-numerics")

        return classifier_loss_final + objectness_loss_final + bounding_box_loss_final
    
    @staticmethod
    def load_and_preprocess_image(image_path):
        im = cv2.imread(image_path)
        X = np.zeros(im.shape, dtype='float32')
        for c in range(3):
            X[:,:,c] = im[:,:,c] / np.amax(im[:,:,c])
        
        return X 
    
    @staticmethod
    def mix_rects(rect1, rect2, alpha=0.5):
        return {
            'left'  : rect1['left']  *(1-alpha) + rect2['left']  *alpha,
            'right' : rect1['right'] *(1-alpha) + rect2['right'] *alpha,
            'top'   : rect1['top']   *(1-alpha) + rect2['top']   *alpha,
            'bottom': rect1['bottom']*(1-alpha) + rect2['bottom']*alpha
        }

    @staticmethod
    def parse_model_output(model_output):
        # Extract non-negligible predictions from model output
        def sigmoid(x):
            return 1 / ( 1 + np.exp(-x) )

        objectness_thres = 0.2
        detections = []
        for i in range(model_output.shape[1]):
            for j in range(model_output.shape[2]):
                if sigmoid(model_output[0, i, j, 62]) > objectness_thres:
                    detection = CaptchaBreaker.decode_rect(model_output[0, i, j, 63:], np.array([j, i], dtype='int'), 8)
                    highest_val = 0
                    highest_ind = 0
                    for l in range(62):
                        if model_output[0, i, j, l] > highest_val:
                            highest_val = model_output[0, i, j, l]
                            highest_ind = l
                    detection['label'] = CaptchaBreaker.label_ind_lookup[highest_ind]
                    detection['objectness'] = float( sigmoid(model_output[0, i, j, 62]) )
                    detections.append( detection )

        # The Plan:
        # * Find all intersections with int'-over-union above a certain "conflict" threshold, and sort by that factor
        # * Search for detections to merge, starting with the strongest intersections
        #   * In order to be merged, detections must agree in label and have intersection strength over some "merge" threshold
        #   * The final detection rect is a combination of the input rects weighted by objectness
        #   * The final objectness satisfies 1-Of == (1-O1)*(1-O2). This means the resultant objectness is always higher than either input, and is highest when both inputs were already high
        #   * The whole process is started over after every merge
        # * If there are no detections to merge, take the strongest remaining conflict, and drop the detection with the lowest objectness
        #   * Restart the whole process after every deletion. We have to re-assess the conflicting intersections as there may not be any now, even if there were multiple before
        # * The process terminates when there are no remaining intersections above the conflict threshold
        conflict_thres = 0.3
        merge_thres    = 0.7
        
        done = False
        while not done:
            done = True
            
            # Get sorted intersections
            intersections = []
            for i in range(len(detections)):
                for j in range(i+1, len(detections)):
                    int_factor = max(CaptchaBreaker.overlap_fraction(detections[i], detections[j]), CaptchaBreaker.overlap_fraction(detections[j], detections[i]))
                    if int_factor >= conflict_thres:
                        intersections.append( (CaptchaBreaker.intersection_over_union(detections[i], detections[j]), i, j) )
            intersections.sort(key=lambda x:x[0], reverse=True)

            # Can we merge any?
            start_over = False
            for int_factor, i, j in intersections:
                if int_factor >= merge_thres and detections[i]['label'] == detections[j]['label']:
                    o1 = detections[i]['objectness']
                    o2 = detections[j]['objectness']
                    epsilon = 0.00001 
                    mix_factor = (1-o1) / (1-o1 + 1-o2 + epsilon) # The farther o1 is from 1 compared to their combined distance from one, the more the resultant will favor detection[j]
                    
                    merged_detection = CaptchaBreaker.mix_rects(detections[i], detections[j], mix_factor)
                    merged_detection['label'] = detections[i]['label']
                    merged_detection['objectness'] = 1 - (1-o1)*(1-o2)
                    detections.append( merged_detection )

                    del detections[j]
                    del detections[i]

                    start_over = True
                    break
            if start_over:
                done = False
                continue

            # Let them fight!
            if len(intersections) > 0:
                done = False
                _, i, j = intersections[0]
                if detections[i]['objectness'] > detections[j]['objectness']:
                    del detections[j]
                else:
                    del detections[i]
        
        return detections