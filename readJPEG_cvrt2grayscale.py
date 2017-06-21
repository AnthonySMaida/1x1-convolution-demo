#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 11:46:00 2017

@author: maida
Filename: readJPEG.py
"""

import matplotlib.pyplot as plt
import numpy as np
import os
#from scipy import ndimage
#from PIL import Image
import tensorflow as tf
import matplotlib.cm as cm

cwd = os.getcwd()
print(cwd)
filename = 'image_0004_leafCropped.jpg'
print(filename)
#im = ndimage.imread(cwd+'/'+filename) # both ways work
im = plt.imread(cwd+'/'+filename, format = 'jpeg')
plt.imshow(im, cmap = cm.Greys_r)
plt.show()
print('Image shape after reading:',im.shape)
print('Image data type:',type(im))
print('Add new dimension for conv2d compatibility.')
im4d = np.expand_dims(im, axis=0)
print('New shape: ', im4d.shape)
#im2 = Image.open(filename) # returns 'PIL.JpegImagePlugin.JpegImageFile'
#print(type(im2))
#plt.imshow(im)

model = tf.Graph()
with model.as_default():
    my_image = tf.constant(im4d, dtype=tf.float32)
    # perform a 1x1 convolution
    # shape = 1 x 1 x 3 x 1
    wts      = tf.constant([[[[0.21], [0.72], [0.07]]]],
                           dtype=tf.float32)
    gray     = tf.nn.conv2d(my_image, wts, [1, 1, 1, 1], padding='SAME')

with tf.Session(graph=model) as sess:
    output = sess.run(gray)
    
print('Output shape after grayscale conversion: ', output.shape)
output.resize((451, 451))
print('Resized for imshow:', output.shape)
#print(output.shape)
print('Print some matrix values to show it is grayscale.')
print(output)
print('Display the grayscale image.')
plt.imshow(output, cmap = cm.Greys_r)
