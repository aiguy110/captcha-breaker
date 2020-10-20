from tensorflow.keras import models 
import tensorflow as tf
import numpy as np
import model
import json
import cv2
import sys

im = cv2.imread(sys.argv[1])
input_data = im[np.newaxis, :,:,:]
input_data = input_data.astype('float32')

my_model = models.load_model('saves/model_v2', custom_objects={'yolo_loss': model.CaptchaBreaker.yolo_loss})
model_output = my_model.predict(input_data)
print(model_output.shape)

best_cell = np.zeros(67, dtype='float32')
best_offset = np.zeros(2, dtype='int')
for i in range(model_output.shape[1]):
    for j in range(model_output.shape[2]):
        if model_output[0, i, j, 62] > best_cell[62]:
            best_cell = model_output[0, i, j, :]
            best_offset = np.array([i, j], dtype='int')

with open('test.json', 'w') as f:
    highest_val = 0
    highest_ind = 0
    for i in range(62):
        if best_cell[i] > highest_val:
            highest_val = best_cell[i]
            highest_ind = i
    print(best_offset)
    print(model.CaptchaBreaker.alphanum[i])
    json.dump( model.CaptchaBreaker.decode_rect(best_cell[63:], best_offset, 8), f )