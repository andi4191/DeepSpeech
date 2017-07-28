#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import tensorflow as tf
import os

from sklearn.cluster import KMeans
import numpy as np

tf.app.flags.DEFINE_float('prune_threshold',    '0.001',    'threshold for pruning')
tf.app.flags.DEFINE_float('lstm_pr_thresh',     '0.001',    'prune threshold for lstm cell parameters')
tf.app.flags.DEFINE_float('bias_pr_thresh',      '0.05',    'prune threshold for bias paranters')
tf.app.flags.DEFINE_string('checkpoint_dir',         '',    'Location for the checkpoint files')
tf.app.flags.DEFINE_boolean('save_checkpoint',    False,    'Save as new checkpoint files')

FLAGS = tf.app.flags.FLAGS


def get_latest_checkpoint(session):

    cmd = "ls -ltr  " + FLAGS.checkpoint_dir + " | grep meta | tail -n 1 | awk '{print $9}'"
    ckpt = os.popen(cmd).read().strip()

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    #Extract from checkpoint filename
    global global_step
    global_step = int(os.path.basename(ckpt_state.model_checkpoint_path).split('-')[1])
    saver = tf.train.import_meta_graph(os.path.join(FLAGS.checkpoint_dir, ckpt))
    saver.restore(session,tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    return saver


def prune(session, saver):


    param_name = ['h1', 'h2', 'h3', 'h5', 'h6', 'b1', 'b2', 'b3', 'b5', 'b6', 'bidirectional_rnn/fw/basic_lstm_cell/weights', 'bidirectional_rnn/bw/basic_lstm_cell/weights']

    pr_thresh = {'h1': FLAGS.prune_threshold,
                 'h2': FLAGS.prune_threshold,
                 'h3': FLAGS.prune_threshold,
                 'h5': FLAGS.prune_threshold,
                 'h6': FLAGS.prune_threshold,
                 'b1': FLAGS.bias_pr_thresh,
                 'b2': FLAGS.bias_pr_thresh,
                 'b3': FLAGS.bias_pr_thresh,
                 'b5': FLAGS.bias_pr_thresh,
                 'b6': FLAGS.bias_pr_thresh,
                 'bidirectional_rnn/fw/basic_lstm_cell/weights': FLAGS.lstm_pr_thresh,
                 'bidirectional_rnn/bw/basic_lstm_cell/weights': FLAGS.lstm_pr_thresh
                 }
    params = {}
    pruned_mask = {}
    prune_thresh = {}
    update_mask = {}

    for p in param_name:

        # Construct the param name as per checkpoint file
        var_name = p+':0'

        # Retrieve the parameter from the checkpoint file
        params[p] = session.graph.get_tensor_by_name(var_name)

        # Create a mask for pruning
        pruned_mask[p] = tf.Variable(tf.ones_like(params[p]))

        # Compute the threshold for pruning for each layer as per the standard deviation
        prune_thresh[p] = tf.sqrt(tf.nn.l2_loss(params[p]))

        # Multiply by the factor of pruning threshold defined
        pr_thresh[p] = pr_thresh[p] * prune_thresh[p]

        # Update the pruning mask as per prune_threshold
        pruned_mask[p] = tf.multiply(pr_thresh[p], pruned_mask[p])

        # Update the paramter after pruning
        update_mask[p] = tf.assign(params[p], tf.multiply(params[p], tf.to_float(tf.greater(tf.abs(params[p]), pruned_mask[p]))))

    # Initialize the variables defined in the graph
    session.run(tf.global_variables_initializer())

    # Apply all the ops
    apply_op = [update_mask[p] for p in param_name]
    update_all_params = tf.group(*apply_op)

    session.run(update_all_params)

    # Calculate the percentage of parameters pruned
    percentage_prune = {}
    for p in param_name:
        initial_size, non_zero = session.run([tf.size(params[p]), tf.count_nonzero(params[p])])
        percentage_prune[p] = 100*(initial_size - non_zero)/initial_size
        print('Param %s with dimensions %s pruned %f %s' % (p, str(params[p].shape, ),  percentage_prune[p], '%'))


    # Save the updated pruned parameters as a latest checkpoint file
    if FLAGS.save_checkpoint:
        saver.save(session, tf.train.latest_checkpoint(FLAGS.checkpoint_dir), global_step=global_step+1)
    else:
        print('Not saving as checkpoint file')


def main(_):

    session = tf.Session()
    saver = get_latest_checkpoint(session)
    prune(session, saver)


if __name__=="__main__":
    tf.app.run()
