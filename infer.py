from tensorflow.keras import models 
import tensorflow as tf
import numpy as np
import model
import json
import cv2
import sys

im = model.CaptchaBreaker.load_and_preprocess_image(sys.argv[1])
input_data = im[np.newaxis, :,:,:]

my_model = models.load_model('saves/model_v3.1', custom_objects={'yolo_loss': model.CaptchaBreaker.yolo_loss})
model_output = my_model.predict(input_data)

objectness_thres = 0.2
detections = []
for i in range(model_output.shape[1]):
    for j in range(model_output.shape[2]):
        if model_output[0, i, j, 62] > objectness_thres:
            detection = model.CaptchaBreaker.decode_rect(model_output[0, i, j, 63:], np.array([j, i], dtype='int'), 8)
            highest_val = 0
            highest_ind = 0
            for l in range(62):
                if model_output[0, i, j, l] > highest_val:
                    highest_val = model_output[0, i, j, l]
                    highest_ind = l
            detection['letter'] = model.CaptchaBreaker.label_ind_lookup[highest_ind]
            detection['objectness'] = float(model_output[0, i, j, 62])
            detections.append( detection )

with open('test.json', 'w') as f:
    json.dump(detections, f)

# best_cell = np.zeros(67, dtype='float32')
# best_offset = np.zeros(2, dtype='int')
# for i in range(model_output.shape[1]):
#     for j in range(model_output.shape[2]):
#         if model_output[0, i, j, 62] > best_cell[62]:
#             best_cell = model_output[0, i, j, :]
#             best_offset = np.array([j, i], dtype='int')

# with open('test.json', 'w') as f:
#     highest_val = 0
#     highest_ind = 0
#     for i in range(62):
#         if best_cell[i] > highest_val:
#             highest_val = best_cell[i]
#             highest_ind = i
#     print(best_offset)
#     print(best_cell[63:])
#     print(model.CaptchaBreaker.alphanum[highest_ind])
#     json.dump( [model.CaptchaBreaker.decode_rect(best_cell[63:], best_offset, 8)], f )