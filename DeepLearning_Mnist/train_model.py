import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import backend as K
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
# input image dimensions
img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], img_cols, img_rows, 1)
    x_test = x_test.reshape(x_test.shape[0], img_cols, img_rows, 1)
    input_shape=(img_cols, img_rows, 1)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train/255
x_test = x_test/255

y_dummy_train = to_categorical(y_train)
y_dummy_test = to_categorical(y_test)


model = Sequential()
model.add(Convolution2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = input_shape , padding = 'same'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))

model.add(Convolution2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(Convolution2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])


hist = model.fit(x_train, y_dummy_train,validation_data = (x_test, y_dummy_test), epochs=15, batch_size=64)

model.save("full_model.mnist")