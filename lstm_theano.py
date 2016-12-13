#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 17:58:34 2016

@author: troywinfree

lstm networks implemented with theano
"""

import numpy as np

import theano
from theano import tensor as T

floatX=theano.config.floatX

# activation functions
sigma_g = T.nnet.sigmoid
sigma_c = T.tanh
sigma_h = lambda x : x

# a single lstm neuron
def lstm_neuron(x_t,c_tm1,h_tm1,W,U,b,n,m) : 
    
    Wx_p_Ux_p_b = W[:4*m,:n].dot(x_t[:n]) + U[:4*m,:m].dot(h_tm1[:m]) + b[:4*m]

    f = sigma_g(Wx_p_Ux_p_b[:m])
    i = sigma_g(Wx_p_Ux_p_b[m:2*m])
    o = sigma_g(Wx_p_Ux_p_b[2*m:3*m])
    s = sigma_c(Wx_p_Ux_p_b[3*m:])

    c = T.zeros_like(c_tm1)
    c = T.set_subtensor(c[:m],f*c_tm1[:m] + i*s)
    
    h = T.zeros_like(h_tm1)
    h = T.set_subtensor(h[:m],o*sigma_h(c[:m]))
    
    return c,h

# stack up the lstm layers
def stack_layer(W,U,b,c0,h0,n,m,x) : 
    
    result,_ = theano.scan(lstm_neuron,
                           sequences = [x],
                           outputs_info = [c0,h0],
                           non_sequences = [W,U,b,n,m])
    
    return result[1]

# declare the theano variables
ms = T.vector('ms',dtype = 'int64')
ns = T.vector('ns',dtype = 'int64')
c0s = T.matrix('c0s',dtype = floatX)
h0s = T.matrix('h0s',dtype = floatX)
Ws = T.tensor3('Ws',dtype = floatX)
Us = T.tensor3('Us',dtype = floatX)
bs = T.matrix('bs',dtype = floatX)
x = T.matrix('x',dtype = floatX)

# tell theano to build the graph
outputs,_ = theano.scan(stack_layer,
                        sequences = [Ws,Us,bs,c0s,h0s,ns,ms],
                        outputs_info = [x])

# tell theano to compile the graph
network = theano.function(inputs = [x,Ws,Us,bs,c0s,h0s,ns,ms],
                          outputs = outputs)

# now a test

np.random.seed(10)

# the number of layers in the network
_n_layers = 5

# the input and output dimensions of each layer
_ns = np.array([np.random.randint(1,5) for i in range(_n_layers)])
_ms = np.concatenate((_ns[1:],[np.random.randint(1,5)]))

# we are padding everything with zeros so we need the max
# input and output dimensions - padding was the only way I 
# could manage to pass varying sized weight matrices to 
# theano's scan function
_max_n = np.max(_ns)
_max_m = np.max(_ms)

# weights
_Ws = np.random.rand(_n_layers,4*_max_m,_max_n)
_Us = np.random.rand(_n_layers,4*_max_m,_max_m)
_bs = np.random.rand(_n_layers,4*_max_m)

# initial cell and output states for each layer
_c0s = np.zeros((_n_layers,_max_m))
_h0s = np.zeros((_n_layers,_max_m))

# the input to the model
_l = 10
_x = np.zeros((_l,np.max([_max_n,_max_m])))
_x[:,:_ns[0]] = np.random.rand(_l,_ns[0])

# compute the value of the network at the input
h_network = network(_x,_Ws,_Us,_bs,_c0s,_h0s,_ns,_ms)

# verify against the by-hand implentation

TOL = 1E-7

from lstm import T_lstm
from activation_functions import identity, logistic, hyptan

# loop over the layers and check the outputs

_test_x = _x[:,:_ns[0]]

for i in range(_n_layers) : 

    lstm = T_lstm(_ns[i],_ms[i],np.zeros(_ms[i]),np.zeros(_ms[i]),
                  logistic(_ms[i]),hyptan(_ms[i]),identity(_ms[i]))
    lstm.W = _Ws[i,:4*_ms[i],:_ns[i]]
    lstm.U = _Us[i,:4*_ms[i],:_ms[i]]
    lstm.b = _bs[i,:4*_ms[i]]
    
    _test_x = lstm(_test_x)
    
    assert np.max(np.fabs(_test_x-h_network[i,:,:_ms[i]])) < TOL, "FAILURE"
