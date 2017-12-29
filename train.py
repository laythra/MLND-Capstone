# We use this file to load our datasets, and train them, notice we have two datasets, one for carans and the other for 
# the non-cars


# Import necessary libraries 
###########################################################################################

import pandas
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import preprocessing
import myModel

from keras.applications import vgg16
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import newaxis

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Conv2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda, Convolution2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint

# 	

###########################################################################################

# Define the paths of of training data, the cars dataset and noncars dataset.

path1 = './datasets/data/train/cars_train/' 
path2 = './datasets/data/train/noncars-train/'

X = []
y = []
files = os.listdir(path1)
i = 0
for file in files:
    X.append(path1+file)
    y.append(1)
    i += 1
    if i == 4500:
        break # Will only use 4500 pictures

files = os.listdir(path2)
for file in files:
    X.append(path2+file)
    y.append(0)

X = np.asarray(X)
y = np.asarray(y)
shuffle(X, y, random_state=2)

X, X_valid, y, y_valid = train_test_split(X, y, train_size=0.8) # Using 20% of our data as validation data

# Uncomment the following section if you would like to view some information about our training test
# print(X.shape)
# print(X_valid.shape)
# print(y.shape)
# print(y_valid.shape)

# Uncomment the following section if you would like to view a sample of the testing set
# img = cv2.imread(X[0])
# plt.imshow(img)
# print(img.shape)

###########################################################################################

HEIGHT = WIDTH = 150
def gen_batches(X, y, batch_size=32): # We generate our data in batches
    X_batch = np.empty([batch_size, HEIGHT, WIDTH, 1])
    y_batch = np.empty(batch_size)
    while True:
        j = 0
        for i in np.random.permutation(X.shape[0]):
            X_batch[j] = preprocessing.preprocess(X[i])
            y_batch[j] = y[i]
            j += 1
            if j == batch_size:
                break
        yield X_batch, y_batch



##########################################################################################

model = myModel.CNN(comp = True, summary = True)

#model = myModel.CNN2(comp = True)
#You're going to have to uncomment the following section if you set 'comp' to False
#LR = 1e-4
#model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])


train = gen_batches(X, y)
valid = gen_batches(X_valid, y_valid)



checkpoint = ModelCheckpoint('sure.h5', 
                             monitor='val_loss', 
                             verbose=1, 
                             save_best_only=True,  
                             period=1)


EPOCHS = 20
history = model.fit_generator(train,
                              samples_per_epoch=len(X),
                              nb_epoch=EPOCHS, 
                              validation_data=valid,
                              nb_val_samples=len(X_valid),
                              verbose = 1,
                              callbacks = [checkpoint]
                            )

model.save('mymodel.h5')

##########################################################################################

