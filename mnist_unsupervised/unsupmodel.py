# *-* encoding utf-8
# Author: kalizar78
from __future__ import print_function
import tensorflow as tf
import numpy as np
import sys
sys.path.append('../tf')
import lstm


rnn_name_enc = 'LSTMEncoder'
rnn_name_dec = 'LSTMDecoder'


class Model(object) :
    def __init__(self, batch_size, indim, celldim, reuse = False) :
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

        self.batch_size = batch_size
        self.indim      = indim
        self.celldim    = celldim
        
        with tf.variable_scope('Encoder', reuse = reuse) as scope:
            self.We = tf.get_variable('We', shape = [indim, celldim],
                                      initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.be = tf.get_variable('be', shape = [celldim],
                                 initializer = tf.constant_initializer(0.0))
        
            self.rnn_enc = lstm.LSTMCell(celldim, celldim, batch_size, rnn_name_enc, reuse = reuse)
            self.rnn_enc_state = self.rnn_enc.get_states(rnn_name_enc) # zero initialized state


        with tf.variable_scope('Decoder', reuse = reuse) as scope:
            self.rnn_dec = lstm.LSTMCell(celldim, celldim, batch_size, rnn_name_dec, reuse = reuse)
            self.rnn_dec_state = self.rnn_dec.get_states(rnn_name_dec)
            self.Wd = tf.get_variable('Wd', shape = [celldim, indim],
                                      initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.bd = tf.get_variable('bd', shape = [indim],
                                 initializer = tf.constant_initializer(0.0))

            

    def encode(self, xt, nsteps, keep_prob) :
        """
        Encoding to cell
        xt : tensor of shape [batch, nsteps, indim]
        returns:
        ct, ht
        ct: cell memories of shape [nsteps, batch, celldim]
        ht: hidden state of shape [nsteps, batch, celldim]
        """
        # unroll and embed
        xt_unroll  = tf.reshape(xt, [self.batch_size * nsteps, self.indim])
        Wxt = tf.nn.sigmoid(tf.matmul(xt_unroll, self.We) + self.be) # [batch * nsteps, celldim]
        Wxt_embed = tf.reshape(Wxt, [self.batch_size, nsteps, self.celldim])
        # spatial temporal projection via LSTM
        ct_enc, ht_enc = self.rnn_enc.inference(Wxt_embed, self.rnn_enc_state)
        return ct_enc, ht_enc


    def decode_step(self, x_ch, t) :
        """
        xt, ct, ht of shape [batch, celldim]
        x_ch[0] = xt 
        x_ch[1] = [ct, ht]
        t       = index, unused
        """
        # 1 step RNN embedding
        xt = x_ch[0]
        ch = x_ch[1]
        c_tp1, h_tp1 = self.rnn_dec.inference_step(xt, ch)

        return [h_tp1, [c_tp1, h_tp1]]

        
    def decode(self, cell, nsteps, keep_prob) :
        """
        Generate from cell memories.
        cell: shape [batch_size, celldim] (encoder memory at last step)
        nsteps : number of steps to embed using rnn_decode
        keep_prob : 1 - dropout prob
        Returns:
        [batch_size, nsteps, indim] output sequence 
        """
        
        # Embed using rnn_decoder with:
        # x0 = zeros
        # c0 = cell (from encoder)
        # h0 = hidden (initialized to 0)

        # NOTE: This may have to be a variable...?
        x0 = tf.zeros_like(self.rnn_dec_state[0])
        
        ht, cht = tf.scan(self.decode_step, 
                          tf.constant(np.arange(nsteps).astype(np.float32).reshape([nsteps,1])),
                          initializer = [x0, [cell, self.rnn_dec_state[1]]])

        # ht has shape [nsteps, batch, celldim]
        # Dropout, then decode last state to
        # sequential logits [nsteps * batch, indim]
        h_drop = tf.nn.dropout(ht, keep_prob)
        slogits = tf.reshape(tf.matmul(tf.reshape(h_drop, [-1, self.celldim]), self.Wd) + self.bd,
                             [nsteps, self.batch_size, self.indim]) 
        img_logits = tf.transpose(slogits, [1,0,2]) # [batch, nsteps, indim] 
        return img_logits



    def encode_decode(self, xt, nsteps, keep_prob_enc, keep_prob_dec) :
        """ Apply encoder, decoder in sequence to xt
        xt : [batch, nsteps, indim]
        nsteps : number of sequence steps
        keep_prob : 1 - dropout prob
        """

        ct_enc, ht_enc = self.encode(xt, nsteps, keep_prob_enc)
        img_logits = self.decode(ct_enc[-1], nsteps, keep_prob_dec)
        return tf.nn.sigmoid(img_logits)

    def sample_step(self, x_ch_s, t) :
        """
        xt, ct, ht of shape [batch, celldim]
        x_ch_s[0] = xt
        x_ch_s[1] = [ct, ht]
        x_ch_s[2] = samples of shape [batch_size, indim]
        t = index, unused
        """
        xt = x_ch_s[0] # input
        ch = x_ch_s[1] # current cell, hidden state
        c_tp1, h_tp1 = self.rnn_dec.inference_step(xt, ch)
        # binary logits of shape [batch, indim]. Each coordinate is a binary logit
        bprobs = tf.nn.sigmoid(tf.matmul(h_tp1, self.Wd) + self.bd) # shape: [batch, indim], probs
        """
        bprobs_reshape = tf.reshape(bprobs, [self.batch_size * self.indim,])
        blogits = tf.stack([tf.constant(1.0) - bprobs_reshape, bprobs_reshape], axis=1) # convert to 2 class logits
        rsamples = tf.multinomial(blogits, 1) # shape [batch*indim, 1] where second dimension is {0,1} sample
        samples = tf.cast(tf.reshape(rsamples, [self.batch_size, self.indim]), tf.float32)
        """
        samples = tf.round(bprobs)
        return [h_tp1, [c_tp1, h_tp1], samples]
        
        
    def sample(self, xt, nsteps) :
        """
        Given initial cell value, sample from output distribution (binary per pixel)
        
        """

        ct_enc, ht_enc = self.encode(xt, nsteps, tf.constant(1.0))
        
        x0 = tf.zeros_like(self.rnn_dec_state[0])
        # Sample using ct_enc[-1], i.e. last set of encoder memories
        ht, cht, st = tf.scan(self.sample_step, 
                              tf.constant(np.arange(nsteps).astype(np.float32).reshape([nsteps,1])),
                              initializer = [x0, [ct_enc[-1], self.rnn_dec_state[1]], tf.zeros([self.batch_size, self.indim])])

        return tf.transpose(st, [1,0,2]) # shape [batch, nsteps, indim]
        
