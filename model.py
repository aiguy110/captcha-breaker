import tensorflow as tf
from tensorflow.keras import layers, models

cnn_base = models.Sequential()
cnn_base.add( layers.Conv2D(32, (3,3), activation='relu' )
cnn_base.add( layers.Conv2D(64, (3,3), activation='relu' )
ccn_base.add( layers.MaxPool2D((2,2)) )
cnn_base.add( layers.Conv2D(64, (3,3), activation='relu' )
cnn_base.add( layers.Conv2D(128, (3,3), activation='relu' )
ccn_base.add( layers.MaxPool2D((2,2)) )
cnn_base.add( layers.Conv2D(128, (3,3), activation='relu' )
cnn_base.add( layers.Conv2D(67, (3,3), activation='relu' ) # 26 letters*2 + 10 digits + 1 objectness + 4 bounding box corrections
