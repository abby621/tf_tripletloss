# -*- coding: utf-8 -*-

import tensorflow as tf

def get_batch_from_list(image_list, batch_size, image_size, crop_size, num_epochs=None):
    filesT = tf.convert_to_tensor(image_list, dtype=tf.string)

    # Makes an input queue and batches
    queue = tf.train.slice_input_producer([filesT], num_epochs=num_epochs, shuffle=False)

    images = read_images_from_disk(queue, image_size, crop_size)

    batch = tf.train.batch([images], batch_size=batch_size, capacity=2000)

    return batch
