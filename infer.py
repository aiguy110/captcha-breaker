import tensorflow.keras
import numpy as np
import model
import cv2
import sys

im = cv2.imread(sys.argv[1])
input_data = im[np.newaxis, :,:,:]
input_data = input_data.astype('float32')

my_model = 
model_output = tf.make_ndarray( my_model.predict(input_data) )