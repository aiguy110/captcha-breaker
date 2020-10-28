from tensorflow.keras import models 
from model import CaptchaBreaker
import tensorflow as tf
import random
import numpy as np
import cv2
import sys


if len(sys.argv) != 3:
    print('Usage: python ./inference_visualizer.py <model_dir> <image_path_list>')
    exit()

model_dir = sys.argv[1]
model = models.load_model(model_dir, custom_objects={'yolo_loss': CaptchaBreaker.yolo_loss})

image_path_list_dir = sys.argv[2]
with open(image_path_list_dir) as f:
    image_paths = list( map(lambda x:x.strip(), f.readlines()) )

def random_color():
    return ( random.choice(range(0, 256)), 
             random.choice(range(0, 256)),
             random.choice(range(0, 256)) )

def rectify(x, N):
    return min(max(0, int(x)), N)

for image_path in image_paths:
    # Load the input image
    im = cv2.imread(image_path)
    input_data = CaptchaBreaker.load_and_preprocess_image( image_path )[np.newaxis, :,:,:]

    # Run the model on it
    model_output = model.predict( input_data )
    
    # Extract predictions from model output
    detections = CaptchaBreaker.parse_model_output( model_output )

    # Draw the predictions on the image and display
    for rect in detections:
        top_left     = (rectify(rect['left'], im.shape[1]) , rectify(rect['top'], im.shape[0]))
        bottom_right = (rectify(rect['right'], im.shape[1]), rectify(rect['bottom'], im.shape[0]))
        cv2.rectangle(im, top_left, bottom_right, random_color(), 1)
    
    cv2.imshow('Prediction', im)
    cv2.waitKey()
    

