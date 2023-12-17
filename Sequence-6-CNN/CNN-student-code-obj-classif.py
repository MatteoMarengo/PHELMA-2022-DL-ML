# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:38:28 2020

@author: capliera
"""
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
 
# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm
 
    # define cnn model
def define_model():
    model = Sequential()
       
   
    # Two convolutionnal blocks with Relu activation functions
    
    # First block : 2 * 32 convolution filters of size 3x3 + 2x2 MaxPooling
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Second block: 2 * 64 convolution filters of size 3x3 + 2x2 MaxPooling
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    # Neural network for classification with one hidden layer
    # flatten to transform the 2D maps into a 1D vector for NN: 
    # => input layer of size 32x32x3 = 3072
    model.add(Flatten())
    #hidden layer with 128 nodes
    model.add(Dense(128, activation='relu'))
    # output layer with 10 nodes because of the 10 classes
    model.add(Dense(10, activation='softmax'))
    
    opt='SGD'
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
 
# plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	
 
# run the whome program for model training and testing
# load dataset
print('Data loading...')
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
print('Data normalization...')
trainX, testX = prep_pixels(trainX, testX)
# define model
model = define_model()

# simple early stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=10)]

# fit model
print('Model training...')
history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_data=(testX, testY), verbose=1)
# evaluate model
print('Performances evaluation')
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
print('Train acc= %.3f' % (train_acc * 100.0))
_, test_acc = model.evaluate(testX, testY, verbose=0)
print('Test_acc= %.3f' % (test_acc * 100.0))
# learning curves
summarize_diagnostics(history)
# save model
model.save('final_model.h5')

