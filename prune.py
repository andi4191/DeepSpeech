#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import DeepSpeech as ds
import os

from sklearn.cluster import KMeans
import numpy as np


pr_thresh = 0.005


def get_latest_checkpoint(session):
    checkpoint_dir = os.path.join(os.path.expanduser('~'),'.local', 'share', 'deepspeech', 'ldc93s1')
    cmd = "ls -ltr  " + checkpoint_dir + " | grep meta | tail -n 1 | awk '{print $9}'"
    ckpt = os.popen(cmd).read().strip()

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)

    #Extract from checkpoint filename
    global global_step
    global_step = int(os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1])
    saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, ckpt))
    saver.restore(session,tf.train.latest_checkpoint(checkpoint_dir))


def prune(session):

    h1 = session.graph.get_tensor_by_name('h1:0')
    h2 = session.graph.get_tensor_by_name('h2:0')
    h3 = session.graph.get_tensor_by_name('h3:0')
    h5 = session.graph.get_tensor_by_name('h5:0')

    fw_w = session.graph.get_tensor_by_name('bidirectional_rnn/fw/basic_lstm_cell/weights:0')
    bw_w = session.graph.get_tensor_by_name('bidirectional_rnn/bw/basic_lstm_cell/weights:0')


    global params
    params={'h1':h1, 'h2':h2, 'h3':h3, 'h5':h5, 'fw_w':fw_w, 'bw_w':bw_w}

    pruned_mask_h1 = tf.Variable(tf.ones_like(h1))
    pruned_mask_h2 = tf.Variable(tf.ones_like(h2))
    pruned_mask_h3 = tf.Variable(tf.ones_like(h3))
    pruned_mask_h5 = tf.Variable(tf.ones_like(h5))

    pruned_mask_fw_w = tf.Variable(tf.ones_like(fw_w))
    pruned_mask_bw_w = tf.Variable(tf.ones_like(bw_w))

    h1_prune = tf.multiply(pruned_mask_h1, h1)
    h2_prune = tf.multiply(pruned_mask_h2, h2)
    h3_prune = tf.multiply(pruned_mask_h3, h3)
    h5_prune = tf.multiply(pruned_mask_h5, h5)
    fw_w_prune = tf.multiply(pruned_mask_fw_w, fw_w)
    bw_w_prune = tf.multiply(pruned_mask_bw_w, bw_w)

    thresh_h1 = pr_thresh
    thresh_h2 = pr_thresh
    thresh_h3 = pr_thresh
    thresh_h5 = pr_thresh
    thresh_fw_w = pr_thresh
    thresh_bw_w = pr_thresh

    mat1 = tf.multiply(pruned_mask_h1, tf.to_float(tf.greater(h1, tf.multiply(pruned_mask_h1, thresh_h1))))
    mat2 = tf.multiply(pruned_mask_h2, tf.to_float(tf.greater(h2, tf.multiply(pruned_mask_h2, thresh_h2))))
    mat3 = tf.multiply(pruned_mask_h3, tf.to_float(tf.greater(h3, tf.multiply(pruned_mask_h3, thresh_h3))))
    mat5 = tf.multiply(pruned_mask_h5, tf.to_float(tf.greater(h5, tf.multiply(pruned_mask_h5, thresh_h5))))
    mat_fw_w = tf.multiply(pruned_mask_fw_w, tf.to_float(tf.greater(fw_w, tf.multiply(pruned_mask_fw_w, thresh_fw_w))))
    mat_bw_w = tf.multiply(pruned_mask_bw_w, tf.to_float(tf.greater(bw_w, tf.multiply(pruned_mask_bw_w, thresh_bw_w))))

    update_h1_mask = tf.assign(pruned_mask_h1, mat1)
    update_h2_mask = tf.assign(pruned_mask_h2, mat2)
    update_h3_mask = tf.assign(pruned_mask_h3, mat3)
    update_h5_mask = tf.assign(pruned_mask_h5, mat5)
    update_fw_w_mask = tf.assign(pruned_mask_fw_w, mat_fw_w)
    update_bw_w_mask = tf.assign(pruned_mask_bw_w, mat_bw_w)

    update_all_mask = tf.group(update_h1_mask, update_h2_mask, update_h3_mask, update_h5_mask, update_fw_w_mask, update_bw_w_mask)

    session.run(tf.global_variables_initializer())

    update_h1 = tf.assign(h1, h1_prune)
    update_h2 = tf.assign(h2, h2_prune)
    update_h3 = tf.assign(h3, h3_prune)
    update_h5 = tf.assign(h5, h5_prune)
    update_fw_w = tf.assign(fw_w, fw_w_prune)
    update_bw_w = tf.assign(bw_w, bw_w_prune)

    update_all_params = tf.group(update_h1, update_h2, update_h3, update_h5, update_fw_w, update_bw_w)
    session.run(update_all_mask)
    session.run(update_all_params)

    percentage_prune = []
    percentage_prune.append(session.run([tf.size(h1), tf.count_nonzero(h1)]))
    percentage_prune.append(session.run([tf.size(h2), tf.count_nonzero(h2)]))
    percentage_prune.append(session.run([tf.size(h3), tf.count_nonzero(h3)]))
    percentage_prune.append(session.run([tf.size(h5), tf.count_nonzero(h5)]))
    percentage_prune.append(session.run([tf.size(fw_w), tf.count_nonzero(fw_w)]))
    percentage_prune.append(session.run([tf.size(bw_w), tf.count_nonzero(bw_w)]))

    print('Dimensionality of the parameters')
    for p in params.keys():
        size = session.run(tf.size(params[p]))
        nz = session.run(tf.count_nonzero(params[p]))
        print(p, params[p].shape, ' Pruned: ', 100*(size - nz)/size, '%')

    # Save the updated checkpoint file now

def get_params():

    # After the model is retrained on pruned parameters. Need to revive checkpoint with updated parameters
    # To do : Revive the checkpoint file and then update the params
    return params


if __name__=="__main__":
    # To do
    # 1. Train the model
    # 2. Collect the variables from the checkpointed files
    # 3. Prune all the variables
    # 4. Retrain the model
    session = tf.Session()
    get_latest_checkpoint(session)
    prune(session)

