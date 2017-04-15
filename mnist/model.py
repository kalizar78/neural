# *-* encoding utf-8
# Author: kalizar78
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../tf')
import lstm

def inference(xt, batch_size, nsteps, indim, celldim, keep_prob, reuse = False) :
    """ setup inference graph
    Args: 
    xt : tensor of shape [batch, nsteps, indim]
    batch_size : batch size
    nsteps     : steps
    indims     : input dimensions
    celldim    : cell / hidden dimensions
    keep_prob  : 1 - dropout probability
    reuse      : whether to create variables

    Returns:
    params, activations
    params = [We, be, rnn, Wp, bp] model parameters
    activations = activation list
    """

    # Embed data
    with tf.variable_scope('embedding', reuse = reuse) as scope:
        We = tf.get_variable('W', shape = [indim, celldim],
                             initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        be = tf.get_variable('b', shape = [celldim],
                             initializer = tf.constant_initializer(0.0))

    # unroll and project
    xt_unroll  = tf.reshape(xt, [-1, indim])
    Wxt = tf.reshape(tf.nn.sigmoid(tf.matmul(xt_unroll, We) + be), [batch_size, nsteps, celldim])

    # spatial temporal projection via LSTM
    rnn = lstm.LSTMCell(celldim, celldim, batch_size, 'MyLSTMCell', reuse = reuse)    
    ht, ct = rnn.inference(Wxt)

    # 10 output classes
    with tf.variable_scope('projection', reuse = reuse) as scope:
        Wp = tf.get_variable('W', shape = [celldim, 10],
                             initializer = tf.contrib.layers.xavier_initializer(uniform = False))
        bp = tf.get_variable('b', shape = [10],
                             initializer = tf.constant_initializer(0.0))

    # First regularize via dropout the last hidden state:
    h_drop = tf.nn.dropout(ht[-1], keep_prob)

    logits = tf.nn.sigmoid(tf.matmul(h_drop, Wp) + bp)

    params = [We, be, rnn, Wp, bp]
    activations = [Wxt, ht, ct, logits]
    return params, activations
