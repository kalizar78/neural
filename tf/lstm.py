# *-* encoding utf-8
# Author: kalizar78
from __future__ import print_function
import tensorflow as tf
import numpy as np

#####################################################################
# LSTM Specific initializations using numpy
#####################################################################
# I forgot where I got these initializations from

def _np_ortho_weights(n,m) :
    """ Returns np orthogonal weighs of size [n,m]
    """
    nm = max(n,m)
    u = np.linalg.svd(np.random.randn(nm,nm))[0]
    return u.astype(np.float32)[:n, :m]

def _stacked_ortho_weights(n,m,copies) :
    """ returns a stack of copies ortho weights, each of size [n,m]
    """

    return np.hstack([_np_ortho_weights(n,m) for _ in range(copies)])

def _bias_init(celldim, nactivations) :
    """ initializes b of shape [nactivations * celldim] np arra
    with all zeros except b[celldim:2*celldim]
    """


    assert(nactivations > 2)
    b = np.zeros(nactivations*celldim, dtype = np.float32)
    b[celldim:2*celldim] = 1.0
    return b




#####################################################################
# LSTM Class
# ambiguous dimension will be the number of steps
# batch_size much be specified
#####################################################################


class LSTMCell(object) :

    def __init__ (self, indim, celldim, batch_size, name, reuse = False) : 
        """
        Basic LSTM cell - only declares the variables
        use inference or inference_step to setup inference graph
        Args:
        indim      : input dimension
        celldim    : cell and hidden state dimension
        batch_size : batch size
        name       : name prefix for variable scope. 
        """


        self.num_activations = 4
        self.name = name
        self.indim = indim
        self.celldim = celldim
        self.batch_size = batch_size


        with tf.variable_scope(name, reuse = reuse) as scope:
            # non recurrent parameters
            self.W = tf.get_variable('W', initializer = _stacked_ortho_weights(indim, celldim, self.num_activations))
            # recurrnt parameters
            self.U = tf.get_variable('U', initializer = _stacked_ortho_weights(celldim,celldim,self.num_activations))

            # bias
            self.b = tf.get_variable('b', initializer = _bias_init(celldim, self.num_activations))

            # hidden state and cell memories:
            self.hstate = tf.get_variable('hstate', initializer = np.zeros((batch_size, celldim), dtype = np.float32))
            self.cell   = tf.get_variable('cell',   initializer = np.zeros((batch_size, celldim), dtype = np.float32))
            
    def inference(self, x_t) :
        """ Sets up inference graph.
        x_t : [batch, nsteps, indim] tensor
        """

        xt = tf.transpose(x_t, [1,0,2]) # [nsteps, batch, indim]
        # unroll since tf doesn't support np.dot(tensor3, tensor2)
        xt_unroll = tf.reshape(xt, [-1, self.indim])
        Wxt = tf.reshape(tf.matmul(xt_unroll, self.W) + self.b, [-1, self.batch_size, self.num_activations * self.celldim])
        # Apply dropout to non-recurrent connections here (future)

        # Use tf.scan to apply recurrent connections

        hstate_t, cell_t = tf.scan(self._step,
                                   Wxt, # these get sliced
                                   initializer = [self.hstate, self.cell]) # accumulator initial values (don't get set by scan


        return hstate_t, cell_t

    def inference_step(self, x_t) :
        """
        inference step, convenience for stepping once so we don't have to enter tf.scan
        x_t : [batch, 1 , indim] tensor
        """
        xt = tf.transpose(x_t, [1,0,2]) # [1, batch, indim]
        # unroll since tf doesn't support np.dot(tensor3, tensor2)
        xt_unroll = tf.reshape(xt, [-1, self.indim])
        Wxt = tf.reshape(tf.matmul(xt_unroll, self.W) + b, [self.batch_size, self.num_activations * self.celldim])
        # Apply dropout to non-recurrent connections here (future)

        # step unit
        h_tp1, c_tp1 = self._step([self.hstate, self.cell], Wxt)

        # keep consistent with inference tensor return shapes
        resize_shape = [1, self.batch_size, self.cell_dim]
        return tf.reshape(h_tp1, resize_shape), tf.reshape(c_tp1, resize_shape)

    def state_update(self, state, cell) :
        """
        returns state and cell assignment ops
        Note: You should use tf.control_dependencies when using assignment ops
        """

        assign_state = tf.assign(self.hstate, state)
        assign_cell  = tf.assign(self.cell, cell)

        return assign_state, assign_cell
    
    def _step(self, hc, Wxt) :
        """ LSTM stepping function
        Wxt will be a slice of Wxt above in inference, so a tensor2 of shape [batch, celldim]
        hc[0] = previous hstate
        hc[1] = previous cell state
        """
        
        h_tm1 = hc[0]
        c_tm1 = hc[1]
        # Pre-activation
        preact = tf.add(Wxt, tf.matmul(h_tm1, self.U)) # add in recurrent contributions
        it = tf.nn.sigmoid(tf.slice(preact, [0, 0],              [self.batch_size, self.celldim]))
        ft = tf.nn.sigmoid(tf.slice(preact, [0, self.celldim],   [self.batch_size, self.celldim]))
        ot = tf.nn.sigmoid(tf.slice(preact, [0, 2*self.celldim], [self.batch_size, self.celldim]))
        gt = tf.nn.tanh   (tf.slice(preact, [0, 3*self.celldim], [self.batch_size, self.celldim]))

        ct = tf.add(tf.multiply(it, gt), tf.multiply(ft, c_tm1))
        ht = tf.multiply(ot, tf.tanh(ct))

        return [ht, ct]


    
