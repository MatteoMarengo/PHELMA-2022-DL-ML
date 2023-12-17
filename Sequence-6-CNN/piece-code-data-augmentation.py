# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:03:15 2021

@author: capliera
"""

from keras.preprocessing.image import ImageDataGenerator

# run the whole process to train and test the model
# load dataset
trainX, trainY, testX, testY = load_dataset()
# prepare pixel data
trainX, testX = prep_pixels(trainX, testX)
# define model
model = define_model()
# create data generator
datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, 
                          horizontal_flip=True)
# prepare iterator
it_train = datagen.flow(trainX, trainY, batch_size=64)
# fit model
steps = int(trainX.shape[0] / 64)
history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=5, 
                           validation_data=(testX, testY), verbose=1)
# evaluate model
print('Performances evaluation')
_, train_acc = model.evaluate(trainX, trainY, verbose=0)
print('Train acc= %.3f' % (train_acc * 100.0))
_, test_acc = model.evaluate(testX, testY, verbose=0)
print('Test_acc= %.3f' % (test_acc * 100.0))

# Save the learnt model
model.save('final_model.h5')