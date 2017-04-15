# *-* encoding utf-8
# Author: kalizar78
from __future__ import print_function

import tensorflow as tf
import numpy as np

import input_data # for reading MNIST data
import model      # model specification
### Data

datadir = '.'
mnist = input_data.read_data_sets(datadir)

### setup computational graph

# Some training hyper-parameters
batch_size = 50
nsteps = 28 
indim  = 28
celldim = 128
lr = 0.01 # learning rate
l1_scale = 0.0001


x = tf.placeholder(tf.float32, [batch_size, 784], name = 'x')
keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
labels = tf.placeholder(tf.int64, [batch_size,], name = 'sparse_labs')   # sparse labels

xt = tf.reshape(x, [batch_size, nsteps, indim])
params, activations = model.inference(xt, batch_size, nsteps, indim, celldim, keep_prob)
logits = activations[-1]
#### Loss
xe_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits))
l1_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(l1_scale), tf.trainable_variables())
total_loss = xe_loss #+ l1_loss

#### Performance Monitoring
ncorrect = tf.equal(labels, tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(ncorrect, tf.float32))


optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)

train_step = optimizer.minimize(total_loss)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run() # initialize all model params

drop_p = 0.5

for i in xrange(1000000) :
    img, lbl = mnist.train.next_batch(batch_size)

    xe, l1, loss, acc, _ = sess.run([xe_loss, l1_loss, total_loss, accuracy, train_step],
                                    feed_dict = {x: img, labels:lbl, keep_prob: 1. - drop_p})

    if (i % 100 == 0) : 
        print('[Step %d] Loss: %f (xe: %f) (l1: %f) (acc: %f) ' % (i, loss, xe, l1, acc))
    

    if (i % 1000 == 0 and i > 0 ) : # occasionally compute validation error
        validation_error = 0.
        for k in xrange(mnist.validation.images.shape[0] / batch_size) :
            vimg, vlbl = mnist.validation.next_batch(batch_size)
            validation_error += sess.run(accuracy, feed_dict = {x: vimg, labels: vlbl, keep_prob: 1.0})
        print('Validation Accuracy: %f' % (validation_error))







