from keras.preprocessing import image
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam as adam
import csv
import pandas
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from skimage.exposure import equalize_adapthist as clahe
import os
from keras.layers import Lambda
from scipy.misc import imresize as imresize
from scipy.misc import imread as imread


colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

data1 = pandas.read_csv('driving_log.csv', names=colnames)

centre_images1 = data1.center.tolist()
left_images1   = data1.left.tolist()
right_images1  = data1.right.tolist()

# Remove the first element:
centre_images1 = centre_images1[1:]
left_images1   = left_images1[1:]
right_images1  = right_images1[1:]

# Add the elements to y value:
y_train1 = data1.steering.tolist()
y_train1 = y_train1[1:] # Remove the first element
y_train1 = np.float32(y_train1) # Convert to float
y_train1 =  y_train1
y_train_all1 = []

for i in range(len(centre_images1)):
    steering_angle_centre = y_train1[i]
    
    delta_steering_ang   = 0.05
    
    steering_angle_left  = steering_angle_centre + delta_steering_ang
    steering_angle_right = steering_angle_centre - delta_steering_ang 
    
    y_train_all1.append(steering_angle_centre)
    y_train_all1.append(steering_angle_left)
    y_train_all1.append(steering_angle_right)

y_train_all1 = np.array(y_train_all1)

print(y_train_all1.shape)

# Get image Data:
x_train1 = []

for i in range(len(centre_images1)):
    centre = imread(centre_images1[i], False, 'RGB')
    left   = imread(left_images1[i][1:], False, 'RGB')
    right  = imread(right_images1[i][1:], False, 'RGB')

    x_train1.append(centre)
    x_train1.append(left)
    x_train1.append(right)

x_train1 = np.array(x_train1)
print(x_train1.shape)

# Re-edit the images:
x_train_edit1 = []

for i in range(x_train1.shape[0]):
    read = x_train1[i, 60:,:,:]
    x_train_edit1.append(read)

x_train_edit1 = np.array(x_train_edit1)
print(x_train_edit1.shape)
del x_train1

x_train_resize1 = []

for i in range(x_train_edit1.shape[0]):
    read = x_train_edit1[i]
    read = imresize(read, (66, 200), interp='bilinear')
    x_train_resize1.append(read)

x_train_resize1 = np.array(x_train_resize1)
print(x_train_resize1.shape)
del x_train_edit1

def DNN():
    # Start
    model = Sequential()
    # set up lambda layer
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(66,200,3)))
    # Apply a 5x5 convolution with 24 output filters on a 31x98 image:
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', name='cnn1')) 
    # Add a 5x5 convolution on top, with 36 output filters:
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', name='cnn2'))
    # Add a 5x5 convolution on top, with 48 output filters:
    model.add(Convolution2D(54, 5, 5, subsample=(2, 2), border_mode='valid', name='cnn3'))
    # Add a 3x3 convolution on top, with 64 output filters:
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', name='cnn4'))
    # Add a 3x3 convolution on top, with 64 output filters:
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid', name='cnn5'))
    # Flatten
    model.add(Flatten())
    # Fully-Connected Layer -1
    model.add(Dense(1164, name='fc1'))
    # Activation
    model.add(Activation('relu'))
    # Fully-Connected Layer -2
    model.add(Dense(100, name='fc2'))
    # Activation
    model.add(Activation('relu'))
    # Fully-Connected Layer -3
    model.add(Dense(50, name='fc3'))
    # Activation
    model.add(Activation('relu'))
    # Dropout
    model.add(Dropout(0.5))
    # Fully-Connected Layer -4
    model.add(Dense(10, name='fc4'))
    # Activation
    model.add(Activation('relu'))
    # Output Layer
    model.add(Dense(1, name='fc5'))
    
    return model

model = DNN()
adm = adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adm, loss='mse')

model.fit(x_train_resize1, y_train_all1, batch_size=256, nb_epoch=5, validation_split=0.01, shuffle=True)

json_string = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(json_string)

print("DNN architecture Saved,")

model.save_weights('model.h5')
print("Weights Saved.")

