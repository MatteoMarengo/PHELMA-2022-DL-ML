# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 14:21:49 2021

@author: capliera
"""


from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

from  tensorflow.keras.callbacks import TensorBoard
import datetime, os

def add_noise(X_train, noise_factor=0.35):

    X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0*X_train.max(), size=X_train.shape) 
    X_train_noisy = np.clip(X_train_noisy, 0., 1.)

    return X_train_noisy

### Import the dataset and display an image

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train = X_train/255.
X_test = X_test/255.

label_class = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

idx = np.random.randint(X_train.shape[0])

plt.imshow(X_train[idx], cmap='gray_r')
plt.title(label_class[y_train[idx]])
plt.show()

# Build a CNN AutoEncoder

input_img = Input(shape=(28, 28, 1))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
#x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
#x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Autoencoder training

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

# Preparing for Tensorboard use
logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

# Note that during the model training, input and output data are the same 
# (cf. autoencoder.fit(X_train, X_train))

training_history = autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=128,
                validation_data=(X_test, X_test),
                callbacks = [tensorboard_callback])

# Model saving

autoencoder.save('vanillaAE.h5')

# Image reconstruction and display

X_pred = autoencoder.predict(X_test)

idx = np.random.randint(X_test.shape[0])

plt.figure()
plt.subplot(121)
plt.imshow(X_test[idx], cmap = plt.cm.gray)
plt.title('original image')
plt.axis('off')
plt.subplot(122)
plt.imshow(X_pred[idx].reshape(X_test.shape[1], X_test.shape[1]), cmap = plt.cm.gray)
plt.title('reconstructed image')
plt.axis('off')


#Building noisy data

### TO BE COMPLETED

X_test_noisy = add_noise(X_test,noise_factor=0.35)

#Reconstructing denoised images and display

### TO BE COMPLETED

X_pred_noisy = autoencoder.predict(X_test_noisy)

idx = np.random.randint(X_test_noisy.shape[0])

plt.figure()
plt.subplot(121)
plt.imshow(X_test_noisy[idx], cmap = plt.cm.gray)
plt.title('original image after noise')
plt.axis('off')
plt.subplot(122)
plt.imshow(X_pred_noisy[idx].reshape(X_test_noisy.shape[1], X_test_noisy.shape[1]), cmap = plt.cm.gray)
plt.title('reconstructed image after noise')
plt.axis('off')



