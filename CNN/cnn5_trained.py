# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:56:39 2020

@author: nannib
from https://sanjayasubedi.com.np/deeplearning/cnn-cat-vs-dog/
require tensorflow 2.2.0 and gast 0.3.3 and tqdm 4.48.2
"""
import matplotlib.pyplot as plt
#import keras
import numpy as np
import glob
import os
from PIL import Image
import tqdm


IMG_DIR = "images/train/"
IM_WIDTH = 128
IM_HEIGHT = 128
LB1="dog"
LB2="cat"
# fill the directory ./images/train with LB1.jpg and LB2.jpg files

onlyfiles_test = next(os.walk("images/"))[2] #images/ is your directory for testing images
oft = len(onlyfiles_test)
print (oft)

def read_test_img(directory, resize_to=(IM_WIDTH, IM_HEIGHT)):
    """
    Reads the test image from the given directory
    :param directory directory from which to read the files
    :param resize_to a tuple of width, height to resize the images
    : returns the image only
    """
    files = glob.glob(directory + "*.jpg")
    images = []
    labels = []
    for f in tqdm.notebook.tqdm(files):
        im = Image.open(f)
        im = im.resize(resize_to)
        im = np.array(im) / 255.0
        im = im.astype("float32")
        images.append(im)
        label = f
        labels.append(label)
             
    return np.array(images), np.array(labels)
# put your testing images in ./images directory 
Xt,yt = read_test_img(directory="images/", resize_to=(IM_WIDTH, IM_HEIGHT))



def plot_images(images, labels):
    n_cols = min(5, len(images))
    n_rows = len(images) // n_cols
    fig = plt.figure(figsize=(8, 8))
 
    for i in range(n_rows * n_cols):
        sp = fig.add_subplot(n_rows, n_cols, i+1)
        plt.axis("off")
        plt.imshow(images[i])
        sp.set_title(labels[i])
    plt.show()
   
def humanize_labels(labels):
    """
    Converts numeric labels to human friendly string labels
    :param labels numpy array of int
    :returns numpy array of human friendly labels
    """
    return np.where(labels == 1, LB1, LB2)
 
plot_images(Xt[:oft], yt[:oft])

from keras.models import load_model

 
fname = "model.hdf5"
model=load_model(fname)

predictions = model.predict(Xt)
predictions = np.where(predictions.flatten() > 0.5, 1, 0)
print (predictions,"----")

plot_images(Xt[:oft], humanize_labels(predictions[:oft]))