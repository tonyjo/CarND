import argparse
import base64
from datetime import datetime
import os
import shutil
import cv2
from keras.models import Model
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers import Input
import keras
from skimage.exposure import equalize_adapthist as clahe
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import model_from_json
import json
from keras.models import load_model
from scipy.misc import imresize as imresize

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

'''
def normalize_data(data):
    norm_data = np.zeros((46, 200, 3), dtype=np.float32)

    rgb = data/255.0
    norm_data = clahe(rgb, kernel_size=None, clip_limit=0.01, nbins=256)

    return norm_data
'''

def normalize_data(data):
    a = -0.5
    b = 0.5
    min_value = 0
    max_value = 255

    norm_rgb = a + ( ( (data - min_value)*(b - a) )/( max_value - min_value ) )
    
    return norm_rgb

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = data["steering_angle"]
        # The current throttle of the car
        throttle = data["throttle"]
        # The current speed of the car
        speed = data["speed"]
        # The current image from the center camera of the car
        imgString = data["image"]
        image = Image.open(BytesIO(base64.b64decode(imgString)))
        image_array = np.asarray(image)
        image_array = image_array[60:,:,:]
        #image_array = normalize_data(image_array)
        image_array = imresize(image_array, (66, 200), interp='bilinear')

        steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
        throttle = 0.15

        print(steering_angle, throttle)

        send_control(steering_angle, throttle)

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    #model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    # Load model
    model = model_from_json(loaded_model_json)
    model.summary()

    # compile the model
    model.compile(optimizer='adam', loss='mse')

    # load weights into new model
    model.load_weights("model.h5", by_name=True)
    print("Loaded model from disk")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
