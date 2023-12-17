# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:30:18 2020

@author: capliera
"""
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# DATA LOADING AND NORMALIZATION

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the images
x_train = (x_train / 255) - 0.5
x_test = (x_test / 255) - 0.5

#Then we convert the y values into one-hot vectors

y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)

num_category=10

# By default, the dimensions of the CNN input data must be 4D
# (training_size, height, width, nber_channels)
# => adding a dimension to the loaded data for the channel dim
# but here grey levl data => nb_channel = 1

x_train = x_train[:,:,:,np.newaxis]
x_test = x_test[:,:,:,np.newaxis]

# NETWORK ARCHITECTURE BUILDING
model = Sequential()
# Two convolutionnal blocks with Relu activation functions
#First block : 32 convolution filters of size 3x3 + 2x2 MaxPooling
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
#Second block: 64 convolution filters of size 3x3 + 2x2 MaxPooling
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Neural network for classification with one hidden layer
#flatten to transform the 2D maps into a 1D vector for NN: 
# => input layer of size 28x28x1 = 784
model.add(Flatten())
#hidden layer with 128 nodes
model.add(Dense(128, activation='relu'))
# output layer with 10 nodes because of the 10 classes
model.add(Dense(num_category, activation='softmax'))


model.summary()

# MODEL COMPILING: choosing loss function, optimizer and 
# metrics for perf evaluation
model.compile(
  optimizer='SGD',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

# MODEL TRAINING with hyper-parameter settings
model_log = model.fit(
  x_train,
  y_train,
  epochs=5,
  batch_size=64,
  verbose=1,
  validation_data=(x_test, y_test)
)
# Name of the available data after model fitting
print(model_log.history.keys())
print('\n')

# MODEL EVALUATION of the test set
score = model.evaluate(
  x_test,
  y_test
)
print('Test loss: %0.2f' %score[0]) 
print('Test accuracy: %0.2f' %score[1])

# plotting the metrics: accuracy curves and model loss curves
fig = plt.figure()
plt.subplot(2,1,1)
plt.plot(model_log.history['accuracy'])
plt.plot(model_log.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')

plt.subplot(2,1,2)
plt.plot(model_log.history['loss'])
plt.plot(model_log.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.tight_layout()


# Save the model to disk.
model.save_weights('model.h5')

# Load the model from disk later using:
model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(x_test[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(y_test[:5]) # [7, 2, 1, 0, 4]