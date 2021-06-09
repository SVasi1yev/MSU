import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
import numpy as np
import time
from tensorflow.keras import backend as K

limit = 0

def my_init(shape, dtype=None, partition_info=None):
    global limit
    temp = K.random_uniform(shape, -limit, limit, dtype=dtype)
    return temp

class pytorch_AlexNet(nn.Module):
    def __init__(self):
        super(pytorch_AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(384, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2 * 2, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 8192),
            nn.ReLU(inplace=True),
            nn.Linear(8192, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=0)

def tf_AlexNet():
    global limit
    model = keras.Sequential()
    limit = (1 / (3 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(192, kernel_size=(3, 3),
            strides=(2,2), activation='relu', padding='valid',
            input_shape=(32, 32, 3), kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2)))
    limit = (1 / (192 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(384, kernel_size=(3, 3),
            strides=(1,1), activation='relu', padding='valid',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2)))
    limit = (1 / (384 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3),
            strides=(1,1), activation='relu', padding='valid',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    limit = (1 / (256 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3),
            strides=(1,1), activation='relu', padding='valid',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3),
            strides=(1,1), activation='relu', padding='valid',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2)))
    model.add(keras.layers.Flatten())
    limit = (1 / 1024) ** 0.5
    model.add(keras.layers.Dense(8192, activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    limit = (1 / 8192) ** 0.5
    model.add(keras.layers.Dense(8192, activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    limit = (1 / 8192) ** 0.5
    model.add(keras.layers.Dense(10, activation='softmax',
            kernel_initializer=my_init,
            bias_initializer=my_init))

    return model

class pytorch_VGG16(nn.Module):
    def __init__(self):
        super(pytorch_VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2),    

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(1),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 2 * 2)
        x = self.classifier(x)
        return F.log_softmax(x, dim=0)

def tf_VGG16():
    global limit
    model = keras.Sequential()
    limit = (1 / (3 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='valid',
            activation='relu', input_shape=(32, 32, 3),
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    limit = (1 / (64 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    limit = (1 / (128 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    limit = (1 / (256 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    limit = (1 / (512 * 3 * 3)) ** 0.5
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.Conv2D(512, kernel_size=(3, 3), padding='valid',
            activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    model.add(keras.layers.ZeroPadding2D(1))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Flatten())
    limit = (1 / (2018 * 3 * 3)) ** 0.5
    model.add(keras.layers.Dense(512, activation='relu',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    limit = (1 / (512 * 3 * 3)) ** 0.5
    model.add(keras.layers.Dense(100, activation='softmax',
            kernel_initializer=my_init,
            bias_initializer=my_init))
    
    return model