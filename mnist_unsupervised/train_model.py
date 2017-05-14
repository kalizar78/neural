# *-* encoding utf-8
# Author: kalizar78
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append('../data')

import input_data # for reading MNIST data
import unsupmodel      # model specification

import matplotlib.pyplot as plt
### Data

datadir = '../data'
mnist = input_data.read_data_sets(datadir)

### setup computational graph

# Some training hyper-parameters
batch_size = 50
nsteps = 28 
indim  = 28
celldim = 64
lr = 0.00001 # learning rate
l1_scale = 0.0001
savedir = './models'




x = tf.placeholder(tf.float32, [batch_size, 784], name = 'x')
t = tf.placeholder(tf.float32, [batch_size, 784], name = 'target')
tt = tf.reshape(t, [batch_size, nsteps, indim])
keep_prob_e = tf.placeholder(tf.float32, name = 'keep_prob_enc')
keep_prob_d = tf.placeholder(tf.float32, name = 'keep_prob_dec')
labels = tf.placeholder(tf.int64, [batch_size,], name = 'sparse_labs')   # sparse labels

xt = tf.reshape(x, [batch_size, nsteps, indim])
model = unsupmodel.Model(batch_size, indim, celldim)
yt = model.encode_decode(xt, nsteps, keep_prob_e, keep_prob_d) # [batch, nsteps, indim]
#### Loss
#print('NOTE: Using MSE Loss... May want to switch to Binary Cross Entropy')

mse_loss = tf.reduce_mean(tf.squared_difference(yt, tt))
xe_loss = -tf.reduce_mean((tt * tf.log(yt )) + ((1.0 - tt) * tf.log(1.0 - yt )))
l1_loss = tf.contrib.layers.apply_regularization(tf.contrib.layers.l1_regularizer(l1_scale), tf.trainable_variables())
total_loss = xe_loss + l1_loss

#### Performance Monitoring
accuracy = tf.reduce_mean(1.0 - tf.abs(tt - yt))

#### Optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate = lr)
#optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_step = optimizer.minimize(total_loss)

#### Tensorboard
mse_summ = tf.summary.scalar('mse_loss', mse_loss)
xe_summ = tf.summary.scalar('xe_loss', xe_loss)
l1_summ = tf.summary.scalar('l1_loss', l1_loss)

train_summ = tf.summary.merge([xe_summ, mse_summ, l1_summ])

#### Session, param init

saver = tf.train.Saver()

sess = tf.InteractiveSession()
tf.global_variables_initializer().run() # initialize all model params

summary_writer = tf.summary.FileWriter('./unsupervised_log', graph = sess.graph)

drop_p = 0.5

for i in xrange(1000000) :

    # experimenting with varying dropout regularizing
    """
    if(i % 100 == 0) :
        drop_p = np.random.rand()
    """    
    img, lbl = mnist.train.next_batch(batch_size)
    # flip images occasionally
    """
    if(np.random.rand() > .5) :
        timg = np.flip(img, axis = 1)
    else:
        timg = img
    """

    xe, l1, loss, acc, _, summ = sess.run([xe_loss, l1_loss, total_loss, accuracy, train_step, train_summ],
                                          feed_dict = {x: img, t: img, keep_prob_e: 1. - drop_p, keep_prob_d: drop_p})

    summary_writer.add_summary(summ, i)
    
    if (i % 100 == 0) : 
        print('[Step %d] Loss: %f (xe: %f) (l1: %f) (acc: %f) ' % (i, loss, xe, l1, acc))
    

    if (i % 1000 == 0 ) : # occasionally compute validation error
        validation_acc = 0.
        niter = mnist.validation.images.shape[0] / batch_size
        for k in xrange(niter) : 
            vimg, vlbl = mnist.validation.next_batch(batch_size)
            validation_acc += sess.run(accuracy, feed_dict = {x: vimg, t: vimg, keep_prob_e: 1.0, keep_prob_d: 1.0})
        validation_acc /= niter
        print('Validation Accuracy: %f' % (validation_acc))
        savefile = os.path.join(savedir, 'umodel')
        saver.save(sess, savefile, global_step = i)


        if(validation_acc >.99) :
            print('Early Stopping condition met!')
            break
        if(validation_acc > .75) :
            testimg = sess.run(yt, feed_dict = {x : vimg, keep_prob_e: 1.0, keep_prob_d: 1.0})
            print(vlbl)
            plt.imshow(testimg[0])
            plt.show()
        
    

