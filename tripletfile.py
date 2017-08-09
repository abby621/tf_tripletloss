# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:49:13 2016

@author: souvenir
"""

import tensorflow as tf
import numpy as np

# global var to avoid reloading mean from disk
meanFile = './models/places365/places365CNN_mean.npy'
meanImage = tf.constant(np.load(meanFile), dtype=tf.float32, shape=[256, 256, 3])

def read_image_list(image_list_file):
    """Reads a .txt file containing image paths of image triplets in
       anchor-positive-negative order
    Args:
       image_list_file: a .txt file with three /path/to/image per line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    files = []
    for line in f:
        ancFile, posFile, negFile = line[:-1].split(' ')
        files.extend([ancFile, posFile, negFile])
    return files

def preprocess_img(img, img_size, crop_size, train_phase=True):
    '''
    pre-process input image:
    Args:
        img: 3-D tensor
        batch_size: tuple
        batch_size: tuple
        train_phase: Bool
    Return:
        distorted_img: 3-D tensor
    '''

    # resize and RGB->BGR (for Caffe model)
    img = tf.image.resize_images(img, img_size)
    img = tf.reverse(img, axis=[-1])

    reshaped_image = tf.cast(img, tf.float32)
    reshaped_image -= meanImage

    if train_phase:
        distorted_img = tf.random_crop(tf.image.random_flip_left_right(reshaped_image), [crop_size[0], crop_size[1], 3])
    else:
        distorted_img = tf.image.crop_to_bounding_box(reshaped_image, 14, 14, crop_size[0], crop_size[1])
    return distorted_img

def read_images_from_disk(input_queue, image_size, crop_size):
    """Consumes a single filename and returns a processed image.
    Args:
      filename_tensor: A scalar string tensor.
    Returns:
      tensors: the decoded image
    """
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    example = preprocess_img(example, image_size, crop_size)
    return example

def get_batch(filename, batch_size, image_size, crop_size, num_epochs=None):
    # Reads paths of images together with their labels
    files = read_image_list(filename)

    filesT = tf.convert_to_tensor(files, dtype=tf.string)

    # Makes an input queue and batches
    queue = tf.train.slice_input_producer([filesT], num_epochs=num_epochs, shuffle=False)

    images = read_images_from_disk(queue, image_size, crop_size)

    batch = tf.train.batch([images], batch_size=batch_size, capacity=2000)

    return batch

def get_batch_from_list(image_list, batch_size, image_size, crop_size, num_epochs=None):
    filesT = tf.convert_to_tensor(image_list)

    # Makes an input queue and batches
    queue = tf.train.slice_input_producer([filesT], num_epochs=num_epochs, shuffle=False)

    images = read_images_from_disk(queue[0], image_size, crop_size)

    batch = tf.train.batch([images], batch_size=batch_size, capacity=2000)

    return batch
