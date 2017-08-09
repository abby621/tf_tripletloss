# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from tripletsetfile import TripletSetData
import os.path
import time
from alexnet import CaffeNetPlaces365


def main():
    filename = 'input2.txt'
    checkpoint_file = '../../output/ckpts/checkpoint-99'
    img_size = [256, 256]
    crop_size = [227, 227]
    featLayer = 'fc7'

    image_batch = tf.placeholder(tf.float32, shape=[None, crop_size[0], crop_size[0], 3])

    print("Preparing network...")
    net = CaffeNetPlaces365({'data': image_batch})
    feat = net.layers[featLayer]

    # Create a saver for writing loading checkpoints.
    saver = tf.train.Saver()

    # tf will consume any GPU it finds on the system. Following lines restrict it to "first" GPU
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0"

    # Create data "batcher"
    data = TripletSetData(filename, img_size, crop_size, False)

    print("Starting session...")
    with tf.Session(config=c) as sess:

        # Here's where we need to load saved weights
        saver.restore(sess, checkpoint_file)

        allFeats = []
        num_iters = len(data.files) / 2 # Total hack: Just make sure to have even # of lines of test examples

        for step in range(num_iters):
            start_time = time.time()
            batch, np = data.getBatch()

            f = sess.run(feat, feed_dict={image_batch: batch})
            allFeats.extend(f)
            duration = time.time() - start_time

            # Write the summaries and print an overview
            if step % 1000 == 0:
                print('Step %d: (%.3f sec)' % (step, duration))

    print(allFeats)
    print(len(allFeats))

if __name__ == "__main__":
    main()
