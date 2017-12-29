# This is where we define the model of our architecture.
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import MaxPooling2D, Dense, Flatten, Reshape, GlobalAveragePooling2D, Conv2D, Dropout

def CNN(input_shapee = (150, 150, 1), comp = False, LR = 1e-4, summary = False):
    
    model = Sequential()
    model.add(Conv2D(32, 3, 3, input_shape=(150, 150, 1), activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))


    if summary:
        model.summary()

    if comp:
        model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])


    return model
    
