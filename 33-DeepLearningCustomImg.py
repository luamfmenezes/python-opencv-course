import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,classification_report
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.layers import Dense, Conv2D, MaxPool2D,Flatten
from keras.utils.np_utils import to_categorical


# Tensor -> N dimensional arrays
# (I,H,W,C) -> (Images, Height,Width,Color)

# Densely connected Neural Network
# Any neuronon is connected in one neuron in the next layer

# Convolution Neural Network
# Each unit is connected to a smaller number of neraby unities in the next layer

 
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# convert train numbers from categories

y_cat_test = to_categorical(y_test,10)
y_cat_train = to_categorical(y_train,10)

single_image = x_train[0]

# nomralize data
x_train = x_train/x_train.max()
x_test = x_test/x_train.max()

# clarify tha we have only one color
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)

model = Sequential()

# Convolutional Layer

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1), activation="relu"))

# Pooling layer

model.add(MaxPool2D(pool_size=(2,2)))

# Flatt - tranform the image into 1 dimension

model.add(Flatten())

# Dense Layer
model.add(Dense(128,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# Train the model

model.fit(x_train,y_cat_train, epochs=2)

# Test
model.evaluate(x_test,y_cat_test)

# Prediction return number, not categories
predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))

