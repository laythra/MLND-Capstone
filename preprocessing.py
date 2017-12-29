
#Where we preprocess our data
import numpy as np
from numpy import newaxis

import cv2

HEIGHT = WIDTH = 150
CHANNELS = 1

def resize(img):
    return cv2.resize(img, (HEIGHT, WIDTH), cv2.INTER_AREA) # Resizing our images to 200x200

def normalize(img): # We use this function to convert our img values to values between 0 and 1
    img = img.astype('float')
    img = img / 255.0
    return img

def preprocess(path):
    img = cv2.imread(path) # Reading our image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Converting it to gray
    img = resize(img)
    img = normalize(img)
    img = img[..., newaxis]
    #print(img.shape)
    return img

def preprocessImage(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = resize(img)
    img = normalize(img)
    img = img[..., newaxis]
    return img
