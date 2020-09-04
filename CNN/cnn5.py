# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 19:56:39 2020

@author: nannib
from https://sanjayasubedi.com.np/deeplearning/cnn-cat-vs-dog/
require tensorflow 2.2.0 and gast 0.3.3
"""
import matplotlib.pyplot as plt
#import keras
import numpy as np
import glob
import os
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split


IMG_DIR = "images/train/"
IM_WIDTH = 128
IM_HEIGHT = 128
LB1="dog"
LB2="cat"
# fill the directory ./images/train with LB1.jpg and LB2.jpg files


def read_test_img(directory, resize_to=(128, 128)):
    """
    Reads the test image from the given directory
    :param directory directory from which to read the files
    :param resize_to a tuple of width, height to resize the images
    : returns the image only
    """
    files = glob.glob(directory + "*.jpg")
    images = []
    for f in tqdm.tqdm_notebook(files):
        im = Image.open(f)
        im = im.resize(resize_to)
        im = np.array(im) / 255.0
        im = im.astype("float32")
        images.append(im)
             
    return np.array(images)
# put x.jpg in ./images directory, x.jpg is your testing image 
Xt = read_test_img(directory="images/", resize_to=(IM_WIDTH, IM_HEIGHT))

def read_images(directory, resize_to=(128, 128)):
    """
    Reads images and labels from the given directory
    :param directory directory from which to read the files
    :param resize_to a tuple of width, height to resize the images
    : returns a tuple of list of images and labels
    """
    files = glob.glob(directory + "*.jpg")
    images = []
    labels = []
    for f in tqdm.tqdm_notebook(files):
        im = Image.open(f)
        im = im.resize(resize_to)
        im = np.array(im) / 255.0
        im = im.astype("float32")
        images.append(im)
       
        label = 1 if LB1 in f.lower() else 0
        labels.append(label)
       
    return np.array(images), np.array(labels)
 
X, y = read_images(directory=IMG_DIR, resize_to=(IM_WIDTH, IM_HEIGHT))

onlyfiles = next(os.walk(IMG_DIR))[2] #img_dir is your directory path as string
of = len(onlyfiles)
print (len(onlyfiles))

# make sure we have OF images if we are reading the full data set.
# Change the number accordingly if you have created a subset


assert len(X) == len(y) == of


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# remove X and y since we don't need them anymore
# otherwise it will just use the memory
del X
del y
print (X_train.shape, X_test.shape)

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
 
plot_images(X_train[:of], humanize_labels(y_train[:of]))
plot_images(Xt, "X")

from keras.layers import Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, MaxPool2D
from keras.models import Model, load_model
 
image_input = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
x = Conv2D(filters=32, kernel_size=7)(image_input)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Conv2D(filters=64, kernel_size=3)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Conv2D(filters=128, kernel_size=3)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = MaxPool2D()(x)
 
x = Flatten()(x)
x = Dense(units=64)(x)
x = Activation("relu")(x)
x = BatchNormalization()(x)
x = Dense(units=1)(x)
x = Activation("sigmoid")(x)

model = Model(inputs=image_input, outputs=x)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#model.summary()

model.fit(X_train, y_train, batch_size=64, epochs=6)

#print(model.metrics_names)
model.evaluate(X_test, y_test, batch_size=128)

predictions = model.predict(Xt)
predictions = np.where(predictions.flatten() > 0.5, 1, 0)
print (predictions,"----")

plot_images(Xt, humanize_labels(predictions))

#fname = "weights_cnn.hdf5"
#model.save(fname, overwrite=True)

#model=load_model(fname)
