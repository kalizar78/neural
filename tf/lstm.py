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
            self.W = tf.get_variable('W', shape = [indim, self.num_activations * celldim], dtype = tf.float32,
                                     initializer = tf.contrib.layers.xavier_initializer())
            # recurrnt parameters
            self.U = tf.get_variable('U', initializer = _stacked_ortho_weights(celldim,celldim,self.num_activations))

            # bias
            self.b = tf.get_variable('b', initializer = _bias_init(celldim, self.num_activations))

            # hidden state and cell memories:
            #self.hstate = tf.get_variable('hstate', initializer = np.zeros((batch_size, celldim), dtype = np.float32))
            #self.cell   = tf.get_variable('cell',   initializer = np.zeros((batch_size, celldim), dtype = np.float32))

    def get_states(self, scope_name, reuse = False, init = None) :
        """
        Gets cell, states.
        scope_name : name of scope
        reuse      : reuse of variable
        init       : initializer (if none zero state is returned)
        Returns:
        [cell_0, hstate_0]
        """
        
        if(init is not None) :
            assert(init.shape == (self.batch_size, self.celldim))
            initializer = init
        else :
            initializer = np.zeros((self.batch_size, self.celldim), dtype = np.float32)

            
        with tf.variable_scope(scope_name, reuse = reuse)  as scope:

            c = tf.get_variable('cell', initializer = initializer)
            h = tf.get_variable('hstate', initializer = initializer)

        return [c,h]
        
    def inference(self, x_t, ch) : 
        """ Sets up inference graph.
        x_t : [batch, nsteps, indim] tensor
        ch[0]  : initial cell
        ch[1]  : initial state
        """

        xt = tf.transpose(x_t, [1,0,2]) # [nsteps, batch, indim]
        # unroll since tf doesn't support np.dot(tensor3, tensor2)
        xt_unroll = tf.reshape(xt, [-1, self.indim])
        Wxt = tf.reshape(tf.matmul(xt_unroll, self.W) + self.b, [-1, self.batch_size, self.num_activations * self.celldim])
        # Apply dropout to non-recurrent connections here (future)

        # Use tf.scan to apply recurrent connections

        cell_t, hstate_t = tf.scan(self._step,
                                   Wxt, # these get sliced
                                   initializer = [ch[0], ch[1]]) # accumulator initial values (don't get set by scan

        return cell_t, hstate_t

    def inference_step(self, xt, cht) : 
        """
        inference step, convenience for stepping once so we don't have to enter tf.scan
        x_t : [batch, indim] tensor
        cht : [seed cell ct, seed state ht]
        ct of shape [batch, celldim]
        ht of shape [batch, celldim]
        """
        # Wxt of shape [batch, celldim]
        Wxt = tf.reshape(tf.matmul(xt, self.W) + self.b,
                         [self.batch_size, self.num_activations * self.celldim])
        # Apply dropout to non-recurrent connections here (future)

        # step unit
        h_tp1, c_tp1 = self._step(cht, Wxt)
        
        return c_tp1, h_tp1

    
    def _step(self, ch, Wxt) :
        """ LSTM stepping function
        Wxt will be a slice of Wxt above in inference, so a tensor2 of shape [batch, num_activations*celldim]
        ch[0] = previous cell
        ch[1] = previous hstate
        """

        c_tm1 = ch[0]        
        h_tm1 = ch[1]

        # Pre-activation
        preact = tf.add(Wxt, tf.matmul(h_tm1, self.U)) # add in recurrent contributions
        it = tf.nn.sigmoid(tf.slice(preact, [0, 0],              [self.batch_size, self.celldim]))
        ft = tf.nn.sigmoid(tf.slice(preact, [0, self.celldim],   [self.batch_size, self.celldim]))
        ot = tf.nn.sigmoid(tf.slice(preact, [0, 2*self.celldim], [self.batch_size, self.celldim]))
        gt = tf.nn.tanh   (tf.slice(preact, [0, 3*self.celldim], [self.batch_size, self.celldim]))

        ct = tf.add(tf.multiply(it, gt), tf.multiply(ft, c_tm1))
        ht = tf.multiply(ot, tf.tanh(ct))

        return [ct, ht]


    
