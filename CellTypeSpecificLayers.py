# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 10:52:31 2016 by emin
"""
from lasagne.layers import Layer
import theano
import theano.tensor as T
import numpy as np 
import lasagne.nonlinearities

class DenseEILayer(Layer):
    """ This defines a dense EI layer."""
    def __init__(self, incoming, ei_ratio, num_units, W=lasagne.init.GlorotNormal(0.99), b=lasagne.init.Constant(0.0), leak=1.0, nonlinearity=lasagne.nonlinearities.rectify, diagonal=True, **kwargs):

        super(DenseEILayer, self).__init__(incoming, **kwargs)
        self.nonlinearity  = (lasagne.nonlinearities.identity if nonlinearity is None else nonlinearity)
        self.num_units     = num_units
        self.leak          = leak

        self.num_inputs    = int(np.prod(self.input_shape[1:]))
        self.num_exc_units = np.ceil(ei_ratio * self.num_inputs)
        self.num_inh_units = np.ceil((1.0 - ei_ratio) * self.num_inputs)

        self.W             = self.add_param(W, (self.num_inputs, self.num_units), name="W")
        
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_units,), name="b", regularizable=False)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, diagonal=True, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a batch of feature vectors.
            input = input.flatten(2)

        D     = np.diag( np.concatenate( (np.ones(self.num_exc_units, dtype=theano.config.floatX),-np.ones(self.num_inh_units, dtype=theano.config.floatX)) ) )
        W_new = T.maximum( self.W, 0.0 )
        #W_new = abs( self.W )

        if diagonal is False:
            W_new[np.diag_indices(self.num_inputs)] = 0.0
            W_new                                   = T.dot(D, W_new)
            W_new                                   = (1.0 - self.leak) * T.eye(self.num_inputs) + self.leak * W_new
        else:
            W_new                                   = T.dot(D, W_new)
            W_new                                   = self.leak * W_new

        activation = T.dot(input, W_new)
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
            
        return self.nonlinearity(activation)