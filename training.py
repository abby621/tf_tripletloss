# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:02:07 2016

@author: souvenir
"""

import tensorflow as tf
from tripletfile import get_batch
import os.path
import time
from alexnet import CaffeNetPlaces365


def main():
    ckpt_dir = './output/ckpts'
    log_dir = './output/logs'
    filename = './inputs/input.txt'
    img_size = [256, 256]
    crop_size = [227, 227]
    batch_size = 90 # Make sure it's divBy3
    num_iters = 100
    summary_iters = 10
    save_iters = 1000
    learning_rate = .0001
    margin = 20
    featLayer = 'fc7'
    
    # Queuing op loads data into input tensor
    image_batch = get_batch(filename, batch_size, img_size, crop_size)

    print("Preparing network...")
    net = CaffeNetPlaces365({'data': image_batch})
    feat = net.layers[featLayer]

    # Features are interleaved A-P-N-A-P-N... in batch
    # Get features and compute loss
    idx = tf.range(0, batch_size, 3)
    ancFeats = tf.gather(feat, idx)
    posFeats = tf.gather(feat, tf.add(idx,1))
    negFeats = tf.gather(feat, tf.add(idx,2))

    dPos = tf.reduce_sum(tf.square(ancFeats - posFeats), 1)
    dNeg = tf.reduce_sum(tf.square(ancFeats - negFeats), 1)

    loss = tf.maximum(0., margin + dPos - dNeg)
    loss = tf.reduce_mean(loss)

    # slightly counterintuitive to not define "init_op" first, but tf vars aren't known until added to graph
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    summary_op = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # tf will consume any GPU it finds on the system. Following lines restrict it to "first" GPU
    c = tf.ConfigProto()
    c.gpu_options.visible_device_list="0"

    print("Starting session...")
    with tf.Session(config=c) as sess:
        sess.run(init_op)

        writer = tf.summary.FileWriter(log_dir, sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)

        net.load('./models/places365/alexnet.npy', sess)
        print("Start training...")
        for step in range(num_iters):
            start_time = time.time()
            _, loss_val, pos_val, neg_val = sess.run([train_op, loss, dPos, dNeg])
            duration = time.time() - start_time

            # Write the summaries and print an overview
            if step % summary_iters == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_val, duration))
                # Update the events file.
                summary_str = sess.run(summary_op)
                writer.add_summary(summary_str, step)
                writer.flush()
#
            # Save a checkpoint
            if (step + 1) % save_iters == 0 or (step + 1) == num_iters:
                print('Saving checkpoint at iteration: %d' % (step))
                checkpoint_file = os.path.join(ckpt_dir, 'checkpoint')
                saver.save(sess, checkpoint_file, global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()
