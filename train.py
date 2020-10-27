from model import CaptchaBreaker
import tensorflow as tf
import numpy as np
import random
import math
import json
import cv2
import sys
import os


def load_dataset(sample_paths):
    # TF likes to convert string it gets its hands on to bytes. Watch out for that. It will break stuff
    for s, sample in enumerate(sample_paths):
        sample_as_list = list(sample)
        for i in range(len(sample_as_list)):
            if type(sample_as_list[i]) == bytes:
                sample_as_list[i] = sample_as_list[i].decode()
        sample_paths[s] = sample_as_list

    # Load the first sample just to get dimensions
    im_path, _, _ = sample_paths[0] 
    im = cv2.imread(im_path)
    N = len(sample_paths)
    
    # Calculate output dimensions
    pooling_summary = [2, 2, 2]
    cell_size = 1 
    h, w, _ = im.shape
    for pooling_factor in pooling_summary:
        cell_size *= pooling_factor
        w = int( math.floor(w / pooling_factor) )
        h = int( math.floor(h / pooling_factor) )
    
    # Load the data
    random.shuffle(sample_paths)
    for n, (image_path, label_path, letters) in enumerate(sample_paths):
        X = CaptchaBreaker.load_and_preprocess_image(image_path)
        Y = np.zeros( (h, w, 67 + 3), dtype='float32' )

        with open(label_path) as f: 
            label_data = json.load(f)
        
        for i in range(h):
            for j in range(w):
                # If the cell prior overlaps the label it overlaps most by over detect_thres, train it to match that label
                # If the cell prior overlaps the label it overlaps most by over no_detect_thres, but less than detect_thres
                # this cell gets a free pass (i.e outputs disregarded; no loss incurred). Otherwise, this cell should be trained 
                # to output an objectness of zero.
                detect_thres    = 0.6
                no_detect_thres = 0.4
                
                cell_offset = np.array([j, i])
                cell_rect   = CaptchaBreaker.decode_rect(Y[i, j, 63:67], cell_offset, cell_size) # The slice we're taking here will always just be 4 zeros, but hopefully this makes it more obvious we're getting the prior for this cell
                
                best_label_rect = None
                best_label_ind = None
                best_label_overlap = 0
                for l, label_rect in enumerate(label_data):
                    label_overlap = CaptchaBreaker.overlap_fraction(cell_rect, label_rect)
                    if label_overlap > best_label_overlap:
                        best_label_overlap = label_overlap
                        best_label_ind, best_label_rect = l, label_rect

                if best_label_overlap > detect_thres and CaptchaBreaker.point_is_inside_rect(CaptchaBreaker.rect_center(label_rect), cell_rect):
                    catagory_ind = CaptchaBreaker.label_ind_lookup.index( label_rect['label'] ) 
                    Y[i, j, catagory_ind] = 1
                    Y[i, j, 62]         = 1
                    Y[i, j, 63:67]      = CaptchaBreaker.encode_rect(best_label_rect, cell_offset, cell_size)
                    Y[i, j, 67:]        = [1, 10, 1]
                elif best_label_overlap > no_detect_thres:
                    Y[i, j, 67:] = [0, 0, 0]
                else:
                    Y[i, j, 62] = 0 # This was already the value there but lets make it explicit...
                    Y[i, j, 68] = 1

        yield X, Y
    
def get_dataset_dims_from_generator(gen, args=[]):
    g = gen(*args)
    return tuple( map(lambda x:x.shape, next(g)) )

def make_train_and_test_datasets(data_dir, validation_split=0.8):
    # Get file locations
    if type(data_dir) == bytes:
        data_dir = data_dir.decode()
    sample_paths = [ (os.path.join(data_dir, sample_path), os.path.join(data_dir, sample_path[:-4]+'.json'), sample_path[:-4]) for sample_path in os.listdir(data_dir) if sample_path[-4:] == '.png' ]
    random.shuffle( sample_paths )
    
    # Create tf.data.Dataset objects
    N_train = int(len(sample_paths) * validation_split)
    train_sample_paths = sample_paths[:N_train]    
    train_dataset = tf.data.Dataset.from_generator(
        load_dataset, 
        args=[train_sample_paths],
        output_types=(tf.float32, tf.float32),
        output_shapes=get_dataset_dims_from_generator(load_dataset, [train_sample_paths]))

    test_sample_paths = sample_paths[N_train:]
    test_dataset = tf.data.Dataset.from_generator(
        load_dataset, 
        args=[test_sample_paths],
        output_types=(tf.float32, tf.float32),
        output_shapes=get_dataset_dims_from_generator(load_dataset, [test_sample_paths]))
    
    # Log paths to images used in training and validation
    with open('train_image_paths.txt', 'w') as f:
        for image_path, _, _ in sample_paths[:N_train]:
            f.write(f'{image_path}\n')

    with open('test_image_paths.txt', 'w') as f:
        for image_path, _, _ in sample_paths[N_train:]:
            f.write(f'{image_path}\n')
        
    return train_dataset, test_dataset


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python ./train.py <data_dir>')

    # Load data
    data_dir = sys.argv[1]
    train_dataset, test_dataset = make_train_and_test_datasets( data_dir )

    # Compile and fit model
    model = CaptchaBreaker()
    model.compile(optimizer='Adam', loss=CaptchaBreaker.yolo_loss)
    model.fit(
        train_dataset.batch(100), 
        epochs=5,
        validation_data=test_dataset.batch(100) )

    # Save :D
    model.save('saves/model_v3.1')