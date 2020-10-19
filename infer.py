import numpy as np
import model
import cv2
import sys

im = cv2.imread(sys.argv[1])
input_data = im[np.newaxis, :,:,:]
input_data = input_data.astype('float')

my_model = model.CaptchaBreaker()
print( my_model.predict(input_data).shape )