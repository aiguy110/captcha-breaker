import tensorflow as tf
import numpy as np
import math
import cv2
import sys
import os


def load_dataset(data_dir, validation_split=0.8):
    # Get file locations
    data_dir = sys.argv[1]
    sample_paths = [ (os.path.join(data_dir, sample_path), os.path.join(data_dir, sample_path[:-4]+'.json')) for sample_path in os.listdir(data_dir) if sample_path[-4:] == '.png']

    # Load the first sample just to get dimensions
    im_path, label_path = sample_paths[0] 
    im = cv2.imread(im_path)
    N = len(sample_paths)
    X = np.zeros( (N,) + im.shape, dtype='float32' )
    
    # Calculate output dimensions
    pooling_summary = [2, 2]
    w, h, _ = im.shape
    for pooling_factor in pooling_summary:
        w = int( math.ceil(w / pooling_factor) )
        h = int( math.ceil(h / pooling_factor) )
    
    # Dis play
    print((w,h))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python ./train.py <data_dir>')

    input_dir = sys.argv[1]
    (X_train, Y_train), (X_test, Y_test) = load_dataset(input_dir)
