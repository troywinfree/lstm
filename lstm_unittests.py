#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:45:43 2016

@author: troywinfree

unit tests for lstm
"""

import unittest

import numpy as np

from activation_functions import identity, logistic, hyptan
from lstm import Univariat_T_lstm, T_lstm


class test_lstm(unittest.TestCase) :
    """
    unit tests for lstm
    """
    
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

