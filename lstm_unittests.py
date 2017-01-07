#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: troywinfree

unit tests for lstm
"""

import unittest
import time

import numpy as np

from activation_functions import identity, logistic, hyptan
from lstm_by_hand import Univariat_T_lstm, T_lstm
from lstm import lstm_model


class test_lstm(unittest.TestCase) :
    """
    unit tests for lstm
    """
    
    @classmethod
    def setUpClass(cls) : 
        """
        initialize the lstm model
        """

        cls.n_layers = 5
        cls.batch_size = 20
        cls.l = 10             # length of recursive chain

        np.random.seed(237)
        
        # the input and output dimensions of each layer
        cls.ns = np.array([np.random.randint(1,5) 
                                for i in range(cls.n_layers)])
        cls.ms = np.concatenate((cls.ns[1:],[np.random.randint(1,5)]))
        
        # the model
        cls.model = lstm_model(cls.ns,cls.ms,seed=102)
        
        # compile the network
        start = time.clock()
        cls.model.compile_network()
        cls.model_network_compile_time = time.clock() - start

        # save the weight matrices
        cls.Ws0 = cls.model.Ws.get_value()
        cls.Us0 = cls.model.Us.get_value()
        cls.bs0 = cls.model.bs.get_value()
        
        # tolerance for numerical optimization
        cls.tol = 1E-2
        
        # learning rate
        cls.eta = 1.
        
        # a single layer model
        cls.nn = 3
        cls.mm = 5

        # one layer model for testing against by-hand gradients
        cls.one_layer_model = lstm_model(np.array([cls.nn]),
                                         np.array([cls.mm]),
                                         seed = 1434)
    
    @classmethod
    def tearDownClass(cls) : 
        """
        print compile times
        """
        
        print('\n')
        
        if hasattr(cls,'model_network_compile_time') : 
            print("Model network compile time: %fs"%cls.model_network_compile_time)
        
        if hasattr(cls,'model_loss_compile_time') : 
            print ("Model loss compile time: %fs"%cls.model_loss_compile_time)
            
        if hasattr(cls,'model_sgd_update_compile_time') : 
            print('Model sgd update compile time %fs'%cls.model_sgd_update_compile_time)
        
        if hasattr(cls,'one_layer_grad_loss_compile_time') : 
            print('One layer grad loss compile time: %fs'%cls.one_layer_grad_loss_compile_time)
            
    def test_theano_vs_by_hand_outputs(self) : 
        """
        compare theano and by hand network evaluations
        """
           
        EPSILON = 1E-7
        
        np.random.seed(9124)
        
        x = np.zeros((self.l,np.max([self.model.max_n,self.model.max_m])))
        x[:,:self.ns[0]] = np.random.rand(self.l,self.ns[0])
        
        test_val = x[:,:self.ns[0]]
        model_val = self.model.network(x)
        
        for i in range(self.n_layers) : 
        
            lstm = T_lstm(self.ns[i],self.ms[i],
                          np.zeros(self.ms[i]),np.zeros(self.ms[i]),
                          logistic(self.ms[i]),
                          hyptan(self.ms[i]),
                          identity(self.ms[i]))
            lstm.W = self.model.Ws.get_value()[i,:4*self.ms[i],:self.ns[i]]
            lstm.U = self.model.Us.get_value()[i,:4*self.ms[i],:self.ms[i]]
            lstm.b = self.model.bs.get_value()[i,:4*self.ms[i]]
            
            test_val = lstm(test_val)
    
            self.assertLess(np.max(np.fabs(test_val-model_val[i,:,:self.ms[i]])),
                            EPSILON)
     
    def test_theano_mse_loss(self) : 
        """
        make sure the mse loss functions is working
        """
        
        EPSILON = 1E-7
        
        np.random.seed(351)

        start = time.clock()
        self.model.compile_loss()
        test_lstm.model_loss_compile_time = time.clock() - start
        
        X = np.random.rand(self.batch_size,
                           self.l,
                           np.max([self.model.max_n,self.model.max_m]))
        Y = np.random.rand(self.batch_size,self.l,self.ms[-1])

        loss = np.mean([np.sum((self.model.network(X[i])[-1][:,:self.ms[-1]] 
                         - Y[i])**2,axis=1) for i in range(self.batch_size)])
         
        self.assertLess(np.abs(self.model.loss(X,Y) - loss),EPSILON)
            
    def test_sgd_update(self) : 
        """ 
        test sgd update function by ensuring the loss decreases for the first
        100 iterations of gradient descent on fixed sample input
        """

        eta = 0.1
        
        np.random.seed(1097)

        start = time.clock()
        self.model.compile_sgd_update()
        test_lstm.model_sgd_update_compile_time = time.clock() - start
        
        X = np.random.rand(self.batch_size,
                           self.l,
                           np.max([self.model.max_n,self.model.max_m]))
        Y = np.random.rand(self.batch_size,self.l,self.ms[-1])
        
        loss = np.inf
        for i in range(100) : 
            [loss_next] = self.model.sgd_update(X,Y,eta)
            self.assertLess(loss_next,loss)
            loss = loss_next
     
    def test_grad_loss(self) : 
        """
        compare the gradient computations of the by hand and theano models
        """
        
        TOL = 1E-5
        
        start = time.clock()
        self.one_layer_model.compile_grad_loss()
        test_lstm.one_layer_grad_loss_compile_time = time.clock() - start
        
        lstm_bh = T_lstm(self.nn,self.mm,np.zeros(self.mm),np.zeros(self.mm),
                         logistic(self.mm),hyptan(self.mm),identity(self.mm))
        lstm_bh.W = self.one_layer_model.Ws.get_value()[0,:4*self.mm,:self.nn]
        lstm_bh.U = self.one_layer_model.Us.get_value()[0,:4*self.mm,:self.mm]
        lstm_bh.b = self.one_layer_model.bs.get_value()[0,:4*self.mm]
        
        x = np.random.rand(self.l,max([self.nn,self.mm]))
        y = np.random.rand(self.l,self.mm)
        
        # scale on the by hand model doesn't match that on the theano model
        grad = lstm_bh.ss_grad(x[:,:self.nn],y)*2./float(self.l)

        dW,dU,db = self.one_layer_model.grad_loss(np.array([x]),np.array([y]))
        
        j = 0
        mm = self.mm
        nn = self.nn
        for i in range(4) : 
        
            i_mm = i*mm
            
            val = np.max(np.fabs(grad[j:j+mm*nn] 
                                 - dW[0,i_mm:i_mm+mm,:nn].flatten()))
            self.assertLess(val,TOL)

            j += mm*nn
            val = np.max(np.fabs(grad[j:j+mm*mm] 
                                 - dU[0,i_mm:i_mm+mm,:mm].flatten()))
            self.assertLess(val,TOL)
            
            j += mm*mm
            val = np.max(np.fabs(grad[j:j+mm] 
                                 - db[0,i_mm:i_mm+mm,].flatten()))
            self.assertLess(val,TOL)
            
            j += mm
        
    def test_univariate_t_lstm(self) :
        """
        compare numerical and analytic derivatives of the univariate
        traditional lstm neuron
        """
        
        EPSILON = 1E-3
        
        np.random.seed(0)
        
        k = 100
        
        xs = np.random.rand(k)
        ys = np.random.rand(k)
        
        lstm = Univariat_T_lstm(0.,0.,
                                logistic(1),hyptan(1),identity(1),
                                seed=0)
        
        grad = lstm.ss_grad(xs,ys)

        grad_i = 0
        for i in range(3) : 
            for flag in ['W','U','b'] : 

                WW = None
                if flag == 'W' : 
                    WW = lstm.W
                elif flag == 'U' : 
                    WW = lstm.U
                else : 
                    WW = lstm.b
                
                num_der = lstm.ss_numerical_derivative(WW[i],1E-7,xs,ys,flag,i)
                
                self.assertLess(np.fabs(num_der - grad[grad_i]),EPSILON)
                
                grad_i += 1
        
    def test_multivariate_t_lstm_num_vs_analytic(self) :
        """
        compare numerical and analytic derivatives of the multivariate
        traditional lstm neuron
        """
        
        EPSILON = 1E-3
        
        np.random.seed(0)
        
        k = 50
        n = 20
        m = 10

        xs = np.random.rand(k,n)
        ys = np.random.rand(k,m)
        
        lstm = T_lstm(n,m,np.zeros(m),np.zeros(m),
                      logistic(m),hyptan(m),identity(m),seed=0)
     
        grad = lstm.ss_grad(xs,ys)

        grad_i = 0
        for i in range(3) : 
            for flag,sz in [['W',n*m],['U',m*m],['b',m]] : 

                for j in range(sz) : 

                    WW = None
                    if flag == 'W' : 
                        WW = lstm.W
                        coord = (i*m+int(j/n),j%n)
                    elif flag == 'U' : 
                        WW = lstm.U
                        coord = (i*m+int(j/m),j%m)
                    else : 
                        WW = lstm.b
                        coord = i*m + j

                    num_der = lstm.ss_numerical_derivative(WW[coord],1E-7,
                                                        xs,ys,flag,coord)
                    
                    self.assertLess(np.fabs(num_der - grad[grad_i]),EPSILON)
                    
                    grad_i += 1
    
    def test_multivariate_t_lstm_full_derivative(self) : 
        """
        compare analytic_derivative result to full derivative result
        """
        
        EPSILON = 1E-12
        
        np.random.seed(0)
        
        k = 100
        n = 20
        m = 18
        
        xs = np.random.rand(k,n)
        ys = np.random.rand(k,m)
        
        lstm = T_lstm(n,m,np.zeros(m),np.zeros(m),
                      logistic(m),hyptan(m),identity(m),seed=0)
     
        # gradient of sum of squares loss
        grad = lstm.ss_grad(xs,ys)

        # full derivative tensor
        D = lstm.full_derivative(xs)
        
        # gradient by multiplication by full derivative tensor
        fd_grad = np.einsum('ij,ij...',(lstm(xs) - ys),D)
        
        # check the results
        
        v = grad - fd_grad
        
        self.assertLess(np.sqrt(v.dot(v)),EPSILON)
                        
    def test_uni_vs_multi_t_lstm(self) :
        """
        compare evaluation and differentiate of univatiate and 
        multivarate traditional neurons 
        """
        
        EPSILON = 1E-17
        
        k = 100
        
        lstm = T_lstm(1,1,np.zeros(1),np.zeros(1),
                      logistic(1),hyptan(1),identity(1),seed=0)
        U_lstm = Univariat_T_lstm(0.,0.,
                                  logistic(1),hyptan(1),identity(1),seed=0)
        
        lstm.W[:,0] = U_lstm.W
        lstm.U[:,0] = U_lstm.U
        lstm.b = U_lstm.b.copy()
        
        xs = np.random.rand(k)
        ys = np.random.rand(k)

        v = lstm(xs.reshape(k,1)).flatten() - U_lstm(xs)
        
        self.assertLess(np.sqrt(v.dot(v)),EPSILON)

        U_grad = U_lstm.ss_grad(xs,ys)
        grad = lstm.ss_grad(xs.reshape(k,1),ys.reshape(k,1))
        
        v = U_grad - grad
        
        self.assertLess(np.sqrt(v.dot(v)),EPSILON)
        

if __name__ == '__main__':
    unittest.main()

