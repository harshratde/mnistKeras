#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 07:21:16 2018

@author: harsh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 23:16:57 2018

@author: harsh
"""

import os

print('getting started')

#PROJECT_NAME = 'mnist_keras_conv_tensorbard'
PROJECT_NAME = ''

PATH = '/home/harsh/Documents/Tutorial/Deeplearning/mnist/code/'+PROJECT_NAME
try:
    os.mkdir(PATH)
except:
    pass
    
os.chdir(PATH)

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.callbacks import TensorBoard

import pickle
import time

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.333)
#sess = tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))

MODEL_NAME = 'mnist-cnn-28x2-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir= 'logs/{}'.format(MODEL_NAME))

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20, batch_size=64,callbacks = [tensorboard],validation_split = 0.1)

#===================================================
#   logging
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc


