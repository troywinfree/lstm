#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: troywinfree

lstm networks implemented with theano
"""

import numpy as np
from scipy.optimize import minimize

import theano
from theano import tensor as T

floatX=theano.config.floatX

# activation functions
sigma_g = T.nnet.sigmoid
sigma_c = T.tanh
sigma_h = lambda x : x


def lstm_neuron(x_t,c_tm1,h_tm1,W,U,b,n,m) :
    """ Computes the output of a single lstm neuron. See
    
        https://en.wikipedia.org/wiki/Long_short-term_memory
    
        Input
        
        x_t    - input vector of dimension at least n
        c_tm1  - previous cell output vector, should be dimension at least m
        h_tm1  - previous neuron output vector, at least dimension m
        W      - the four weight matrices for x_t stacked on top of each other
                 (in the order, from top to bottom, forget, intput, output, and
                 cell), should have dimension at least 4*m x n
        U      - the four weight matrices for h_tm1 stacked on top of each 
                 other (same order as W), should have dimension at least 4*m x m
        b      - the bias weight vectors again stacked on top of each other,
                 should have dimension at least 4*m
        n      - dimension of the lstm input
        m      - dimension of the lstm output
        
        Output
        
        c      - cell state
        h      - output
    """
    
    # do the linear algebra
    Wx_p_Ux_p_b = W[:4*m,:n].dot(x_t[:n]) + U[:4*m,:m].dot(h_tm1[:m]) + b[:4*m]

    # compute the activations
    f = sigma_g(Wx_p_Ux_p_b[:m])
    i = sigma_g(Wx_p_Ux_p_b[m:2*m])
    o = sigma_g(Wx_p_Ux_p_b[2*m:3*m])
    s = sigma_c(Wx_p_Ux_p_b[3*m:])

    # compute the cell state
    c = T.zeros_like(c_tm1)
    c = T.set_subtensor(c[:m],f*c_tm1[:m] + i*s)
    
    # compute the output
    h = T.zeros_like(h_tm1)
    h = T.set_subtensor(h[:m],o*sigma_h(c[:m]))
    
    return c,h

    
def lstm_layer(W,U,b,c0,h0,n,m,x) : 
    """ Build a recursive layer of lstm neurons
    
        Input
        
        W      - the four weight matrices for x_t stacked on top of each other
                 (in the order, from top to bottom, forget, intput, output, and
                 cell), should have dimension at least 4*m x n
        U      - the four weight matrices for h_tm1 stacked on top of each 
                 other (same order as W), should have dimension at least 4*m x m
        b      - the bias weight vectors again stacked on top of each other,
                 should have dimension at least 4*m
        c0     - initial cell state vector, at least dimension m 
        h0     - initial output vector, at least dimension m
        n      - dimension of the lstm input
        m      - dimension of the lstm output
        x      - matrix of inputs, number of columns at least n
        
        Output
        
        uncompiled theano symbolic representation of the lstm layer
    
    """
    
    result,_ = theano.scan(lstm_neuron,
                           sequences = [x],
                           outputs_info = [c0,h0],
                           non_sequences = [W,U,b,n,m])
    
    return result[1]


def compute_mean_log_lklyhd_outputs(X,Y,Ws,Us,bs,c0s,h0s,ns,ms) : 
    """ Builds theano symbolic representation of the mean log likelyhood
        loss function.
        
        The weight matrices and bias vectors are padded with zeros
        
        • Ws has dimension ? x 4*max(ms) x max(nx)
        • Us has dimension ? x 4*max(ms) x max(ms)
        • bs has dimension ? x 4*max(ms)
        
        The input tensor X is also padded with zeros
        
        • X has dimension ? x ?? x max(max(ms),max(ns))
        
        Input
    
        X    - domain inputs as 3D tensor
        Y    - range outputs as 3D tensor
        Ws   - input weight matrices as 3D tensor (one matrix per layer)
        Us   - previous output weight matrices as 3D tensor
        bs   - bias vectors as matrix
        c0s  - initial cell state vectors as matrix
        h0s  - initial neuron output vectors as matrix
        ns   - input dimensions for each layer
        ms   - ouput dimensions for each layer
        
        Output
        
        uncompiled symbolic theano representation of mean log likelyhood loss
    """
    
    # the function to scan over - it just collects the log likelyhood of 
    # the positions indicated by y given x and the weight matrices
    def scan_f(x,y,W,U,b,c0,h0,n,m) : 
        
        outs,_ = theano.scan(lstm_layer,
                             sequences = [W,U,b,c0,h0,n,m],
                             outputs_info = [x])
        
        # y is zero except for one one in each row so it's ok to
        # multiply then sum then take the log
        return T.log(T.sum(outs[-1][:,:m[-1]]*y,axis=1)) 
    
    batch_outputs,_ = theano.scan(scan_f,
                                  sequences = [X,Y],
                                  non_sequences = [Ws,Us,bs,c0s,h0s,ns,ms])
    
    return T.mean(batch_outputs)

    
def compute_mean_squared_outputs(X,Y,Ws,Us,bs,c0s,h0s,ns,ms) : 
    """ Builds theano symbolic representation of the mean squared
        loss function 
        
        The weight matrices and bias vectors are padded with zeros
        
        • Ws has dimension ? x 4*max(ms) x max(nx)
        • Us has dimension ? x 4*max(ms) x max(ms)
        • bs has dimension ? x 4*max(ms)
        • c0s has dimension ? x max(ms)
        • h0s has dimension ? x max(ms)
        
        The input tensor X is also padded with zeros
        
        • X has dimension ? x ?? x max(max(ms),max(ns))
        
        Input
    
        X    - domain inputs as 3D tensor
        Y    - range outputs as 3D tensor
        Ws   - input weight matrices as 3D tensor (one matrix per layer)
        Us   - previous output weight matrices as 3D tensor
        bs   - bias vectors as matrix
        c0s  - initial cell state vectors as matrix
        h0s  - initial neuron output vectors as matrix
        ns   - input dimensions for each layer
        ms   - ouput dimensions for each layer
        
        Output
        
        uncompiled symbolic theano representation of mean squared loss
    """
    
    # the function to scan over
    def scan_f(x,y,W,U,b,c0,h0,n,m) : 
        
        outs,_ = theano.scan(lstm_layer,
                             sequences = [W,U,b,c0,h0,n,m],
                             outputs_info = [x])
        
        return T.sum((outs[-1][:,:m[-1]] - y)**2,axis=1)
    
    batch_outputs,_ = theano.scan(scan_f,
                                  sequences = [X,Y],
                                  non_sequences = [Ws,Us,bs,c0s,h0s,ns,ms])
    
    return T.mean(batch_outputs)
    

class lstm_model : 
    """ LSTM neural neetwork. See
        
        https://en.wikipedia.org/wiki/Long_short-term_memory
        
        or
        
        http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        
        It is likely that I am overdoing it with the theano.scan function
    """
    
    def __init__(self,
                 ns,
                 ms,
                 seed = None,
                 c0s = None,
                 h0s = None
                 ) : 
        """ Model initializer
        
            When provided, c0s and h0s need to be padded with zeros
                
            • c0s has dimension len(ns) x max(ms)
            • h0s has dimension len(ns) x max(ms)
            
            note also that len(ns) == len(ms)
        
            Inputs
            
            ns   - input dimensions for each layer
            ms   - ouput dimensions for each layer
            seed - seed for random initialization of weight matrices
            c0s  - optional initial cell state vectors as matrix
            h0s  - optional initial neuron output vectors as matrix    
        """
        
        n_layers = len(ns)
        assert n_layers == len(ms)
        
        assert False not in (ns[1:] == ms[:-1])
        
        self.n_layers = n_layers
        
        self.seed = seed

        # seed the random generator if requested
        if seed is not None : 
            np.random.seed(seed)
        
        # we are padding everything with zeros so we need the max
        # input and output dimensions - padding was the only way I 
        # could manage to pass varying sized weight matrices to 
        # theano's scan function
        self.max_n = np.max(ns)
        self.max_m = np.max(ms)
            
        # collect c0 and h0
        if c0s is None : 
            self.c0s = theano.shared(np.zeros((self.n_layers,
                                              self.max_m)),name='c0s')
        else : 
            # check input
            assert c0s.shape == (self.n_layers,self.max_m)
            
            self.c0s = theano.shared(np.array(c0s),name='c0')
        if h0s is None : 
            self.h0s = theano.shared(np.zeros((self.n_layers,
                                              self.max_m)),name='h0s')
        else : 
            # check input
            assert h0s.shape == (self.n_layers,self.max_m)
            self.h0s = theano.shared(np.array(h0s),name='h0')
              
        # weights
        Ws = np.zeros((self.n_layers,4*self.max_m,self.max_n))
        Us = np.zeros((self.n_layers,4*self.max_m,self.max_m))
        bs = np.zeros((self.n_layers,4*self.max_m))
        
        # we are paadding with zeros
        for i in range(self.n_layers) : 
            Ws[i,:4*ms[i],:ns[i]] = np.random.rand(4*ms[i],ns[i])
            Us[i,:4*ms[i],:ms[i]] = np.random.rand(4*ms[i],ms[i])
            bs[i,:4*ms[i]] = np.random.rand(4*ms[i])
        
        # set the shared variables
        self.Ws = theano.shared(Ws,name='Ws')
        self.Us = theano.shared(Us,name='Us')
        self.bs = theano.shared(bs,name='bs')
        
        # input and output dimensions
        self.ms = theano.shared(ms,name='ms')
        self.ns = theano.shared(ns,name='ns')
         
        # compute the network
        self._compute_network_model()
        
    def _compute_network_model(self) : 
        """ Build the network, loss, grad_loss and sgd_update theano functions.
            More work than is strictly nessecary is done here as the only thing
            that is really needed in order to run sgd (stochastic gradient 
            descent) is the sgd_update function. The network, loss and grad_loss
            functions are compiled since this is experimental code.
        """
        
        # build the network
        self.x = T.matrix('x',dtype = floatX)

        self.network_outputs,_ = theano.scan(lstm_layer,
                                             sequences = [self.Ws,
                                                          self.Us,
                                                          self.bs,
                                                          self.c0s,
                                                          self.h0s,
                                                          self.ns,
                                                          self.ms],
                                             outputs_info = [self.x])

        # build mean squared loss
        
        # the samples are provided as a tensor to support batching of SGD
        self.X = T.tensor3('X',dtype = floatX)
        self.Y = T.tensor3('Y',dtype = floatX)
        
        self.loss_outputs = compute_mean_squared_outputs(self.X,self.Y,
                                                         self.Ws,self.Us,
                                                         self.bs,self.c0s,
                                                         self.h0s,self.ns,
                                                         self.ms)
      
        # get the gradient of the loss
        
        (self.dWs,
         self.dUs,
         self.dbs) = theano.grad(self.loss_outputs,
                                 [self.Ws,self.Us,self.bs])

        # get the sgd updates
        
        # this is the learning parameter
        self.eta = T.scalar('eta',dtype = floatX)
        
        self.sgd_updates = ((self.Ws,self.Ws - self.eta*self.dWs),
                            (self.Us,self.Us - self.eta*self.dUs),
                            (self.bs,self.bs - self.eta*self.dbs))
        
        # set functions to None
        self.network = None
        self.loss = None
        self.grad_loss = None
        self.sgd_update = None

    def compile_network(self) : 
        """ Compile the network
        """
        
        if self.network is not None : 
            return
        
        # note that the return from this is the outputs from each layer
        # also x must have column dimension max(self.max_n,self.max_m)
        self.network = theano.function(inputs = [self.x],
                                       outputs = self.network_outputs)
        
    def compile_loss(self) : 
        """ Compile mse loss
        """
        
        if self.loss is not None : 
            return
        
        self.loss = theano.function(inputs = [self.X,self.Y],
                                    outputs = self.loss_outputs)
        
    def compile_grad_loss(self) : 
        """ Compile the gradient of the loss
        """
        
        if self.grad_loss is not None : 
            return
        
        self.grad_loss = theano.function(inputs = [self.X,self.Y],
                                         outputs = [self.dWs,
                                                    self.dUs,
                                                    self.dbs])
        
    def compile_sgd_update(self) : 
        """ Compile the SGD update
        """
        
        if self.sgd_update is not None : 
            return
        
        # note that this returns the PREVIOUS loss value - using 
        # the weights before the update
        self.sgd_update = theano.function(inputs = [self.X,self.Y,self.eta],
                                          outputs = [self.loss_outputs],
                                          updates = self.sgd_updates)
        
        
    def batch_optimize(self,_X,_Y,tol = 1E-5) : 
        """ Optimize the model using BFGS. This is only for toy problems and 
            serves as a reality check on stochastic gradient descent
            
            Input
            
            _X       - 3D tensor of domain samples
            _Y       - 3D tensor of range samples
            tol      - tolerance for the optimizer
            
            Output
            
            opt_res  - optimization result from the scipy's minimize function
            Ws_opt   - optimal input weights tensor weights
            Us_opt   - optimal previous output weights tensor weights
            bs_opt   - optimal bias weights matrix weights
        """
        
        # set up the theano functions for evaluating the objective and jacobian
        # for the optimization - if I was a bit more serious about this code
        # I would factor this out as it is lengthy, but since it is just 
        # validation code I have left it as-is
        
        Ws = T.tensor3('Ws_bo',dtype = floatX)
        Us = T.tensor3('Us_bo',dtype = floatX)
        bs = T.matrix('bs_bo',dtype = floatX)

        X = theano.shared(_X,'X_bo')
        Y = theano.shared(_Y,'X_bo')
        
        loss_outputs = compute_mean_squared_outputs(X,Y,Ws,Us,bs,
                                                    self.c0s,self.h0s,
                                                    self.ns,self.ms)
        
        loss = theano.function(inputs = [Ws,Us,bs],
                               outputs = loss_outputs)
        
        (dWs,dUs,dbs) = theano.grad(loss_outputs,[Ws,Us,bs])
        
        grad_loss = theano.function(inputs = [Ws,Us,bs],
                                    outputs = [dWs,dUs,dbs])
        
        # function for assembling and disassembling matrix weights
        # from flattened weights
        
        def disassemble(x,ms,ns) : 
            
            Ws = np.zeros((self.n_layers,4*self.max_m,self.max_n))
            Us = np.zeros((self.n_layers,4*self.max_m,self.max_m))
            bs = np.zeros((self.n_layers,4*self.max_m))
             
            nn_0 = 0            
            
            for i in range(self.n_layers) : 
                
                nn_1 = nn_0+4*ms[i]*ns[i]
                nn_2 = nn_1+4*ms[i]*ms[i]

                Ws[i,:4*ms[i],:ns[i]] = x[nn_0:nn_1].reshape(4*ms[i],ns[i])
                Us[i,:4*ms[i],:ms[i]] = x[nn_1:nn_2].reshape(4*ms[i],ms[i])
                
                nn_0 = nn_2 + 4*ms[i]
                bs[i,:4*ms[i]] = x[nn_2:nn_0]
                
            return (Ws,Us,bs)
            
        def assemble(Ws,Us,bs,ms,ns) : 
            
            result = np.array([])

            for i in range(self.n_layers) : 

                Dx = np.concatenate((Ws[i,:4*ms[i],:ns[i]].flatten(),
                                     Us[i,:4*ms[i],:ms[i]].flatten(),
                                     bs[i,:4*ms[i]].flatten()))
                
                result = np.concatenate((result,Dx))
            
            return result
            
        # define the objective an jacobian
        
        def objective(x,ms,ns) : 
            
            return loss(*disassemble(x,ms,ns))
            
        def jac(x,ms,ns) : 
            
            dWs,dUs,dbs = grad_loss(*disassemble(x,ms,ns))
            
            return assemble(dWs,dUs,dbs,ms,ns)
            
        # run the optimization
            
        ms = self.ms.get_value()
        ns = self.ns.get_value()

        x0 = assemble(self.Ws.get_value(),
                      self.Us.get_value(),
                      self.bs.get_value(),
                      ms,ns)

        opt_res = minimize(objective,
                           x0,
                           (ms,ns),
                           'L-BFGS-B',
                           jac,
                           tol = tol)
        
        # collect the optimal weights
        
        Ws_opt,Us_opt,bs_opt = disassemble(opt_res.x,ms,ns)

        return [opt_res, Ws_opt, Us_opt, bs_opt]
