# import model
import numpy as np
import argparse
import preprocessing
import cv2
from numpy import newaxis, expand_dims, transpose
from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D, Dropout
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint



ap = argparse.ArgumentParser()
ap.add_argument("-img", "--image", required = True, help = "Path to the image you would like to classify")
args = vars(ap.parse_args())

myImage = cv2.imread(args["image"])
myImage = np.array(myImage)

from keras.models import load_model
model = load_model('mymodel.h5') # We load the model i already trained using our datasets (cars and non-cars).

myImage = preprocessing.preprocessImage(myImage)
myImage = np.expand_dims(myImage, axis=0)
pred1 = model.predict(np.asarray(myImage), batch_size=1)
print("The probability that there is a car in this picture is: ", pred1[0]*100)