# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 10:48:15 2020

@author: capliera
"""

# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import skimage.io as skio
import matplotlib.pyplot as plt
 
# load and prepare the image
def load_image(filename):
	# load the image
    img = load_img(filename, target_size=(32, 32))
	# convert to array
    img = img_to_array(img)
	# reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
	# prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img
 

# load the image
img = load_image('sample_image.png')
# load model
model = load_model('final_model.h5')
# predict the class
result = model.predict_classes(img)
print(result[0])

img = skio.imread('sample_image.png')
plt.figure()
skio.imshow(img)
 
