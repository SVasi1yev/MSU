import torch as torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import numpy as np
import time
import sys
import os
from models import *

device = sys.argv[1]
framework = sys.argv[2]
model_name = sys.argv[3]
epochs = int(sys.argv[4])
batch_size = int(sys.argv[5])
optimizer_name = sys.argv[6]
learning_rate = float(sys.argv[7])

if framework == 'pytorch':
    if model_name == 'alexnet':
        model = pytorch_AlexNet().to(device)
        (x_train, y_train), (x_test, y_test) = \
                keras.datasets.cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
        temp = np.empty((50000, 3, 32, 32))
        for i in range(3):
            temp[:,i,:,:] = x_train[:,:,:,i]
        x_train = temp
        x_train = torch.Tensor(x_train)
        temp = y_train.argmax(axis=1)
        y_train = temp
        y_train = torch.LongTensor(y_train)
    elif model_name == 'vgg':
        model = pytorch_VGG16().to(device)
        (x_train, y_train), (x_test, y_test) = \
                keras.datasets.cifar100.load_data()
        y_train = keras.utils.to_categorical(y_train, 100)
        y_test = keras.utils.to_categorical(y_test, 100)
        temp = np.empty((50000, 3, 32, 32))
        for i in range(3):
            temp[:,i,:,:] = x_train[:,:,:,i]
        x_train = temp
        x_train = torch.Tensor(x_train)
        temp = y_train.argmax(axis=1)
        y_train = temp
        y_train = torch.LongTensor(y_train)
    else:
        raise ValueError

    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError

    loss_func = nn.CrossEntropyLoss()

    times = []
    batch_num = int(x_train.shape[0] / batch_size)
    for epoch in range(epochs):
        print('Epoch {}'.format(epoch + 1))
        epoch_loss = 0
        epoch_acc = 0
        epoch_time = time.perf_counter()
        for i in range(batch_num):
            start = i * batch_size
            data = x_train[start:min(start + batch_size, x_train.shape[0])]
            target = y_train[start:min(start + batch_size, y_train.shape[0])]
            data = Variable(data).to(device)
            target = Variable(target).to(device)
            out = model(data)
            loss = loss_func(out, target)
            epoch_loss += loss.item()
            pred = out.data.max(1)[1]
            correct_num = (pred == target).sum()
            epoch_acc += correct_num.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        epoch_time = time.perf_counter() - epoch_time
        times.append(epoch_time)
        epoch_acc /= x_train.shape[0]
        epoch_loss /= batch_num
        print('Epoch time: {:.5f}'.format(epoch_time))
        print('Accuracy: {:.5f}'.format(epoch_acc))
        print('Loss: {:.5f}'.format(epoch_loss), end='\n\n')
    avr_time = sum(times[1:]) / (len(times) - 1)
    avr_thrput = 50000 / avr_time
    print('Train finish')
    print('Train time: {:.5f}'.format(sum(times)))
    print('Average epoch time: {:.5f}'.format(avr_time))
    print('Average throughput: {:.5f}'.format(avr_thrput))
    
elif framework == 'tf':
    times = []
    count = 0
    epoch_time = 0

    class MyCallBack(keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs={}):
            global epoch_time
            print('Epoch {}'.format(epoch + 1))
            epoch_time = time.perf_counter()

        def on_epoch_end(self, epoch, logs={}):
            global epoch_time
            epoch_time = time.perf_counter() - epoch_time
            times.append(epoch_time)
            print('Epoch time: {:.5f}'.format(epoch_time))
            print('Accuracy: {:.5f}'.format(logs['acc']))
            print('Loss: {:.5f}'.format(logs['loss']), end='\n\n')
        
        def on_train_end(self, logs={}):
            avr_time = sum(times[1:]) / (len(times) - 1)
            avr_thrput = 50000 / avr_time
            print('Train finish')
            print('Train time: {:.5f}'.format(sum(times)))
            print('Average epoch time: {:.5f}'.format(avr_time))
            print('Average throughput: {:.5f}'.format(avr_thrput))

    if device == "cuda":
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
    elif device == "cpu":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if model_name == 'alexnet':
        model = tf_AlexNet()
        (x_train, y_train), (x_test, y_test) = \
                keras.datasets.cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)
    elif model_name == 'vgg':
        model = tf_VGG16()
        (x_train, y_train), (x_test, y_test) = \
                keras.datasets.cifar100.load_data()
        y_train = keras.utils.to_categorical(y_train, 100)
        y_test = keras.utils.to_categorical(y_test, 100)
    else:
        raise ValueError
    
    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(lr=learning_rate)
    elif optimizer_name == 'sgd':
        optimizer = optimizers.SGD(lr=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=learning_rate)
    else:
        raise ValueError
    
    model.compile(optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['acc'])

    history = MyCallBack()
    model.fit(x_train, y_train, epochs=epochs,
            batch_size=batch_size, shuffle=False,
            verbose=0, callbacks=[history])
else:
    raise ValueError