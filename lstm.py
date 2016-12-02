#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:26:01 2016

@author: troywinfree

lstm neural networks
"""

import numpy as np
from scipy.misc import derivative as n_der


class Univariat_T_lstm : 
    """
    Univariate lstm model for derivative testing
    """
    
    def __init__(self,h0,c0,sigma_g,sigma_c,sigma_h,seed=None) :
        """
        U_lstm initializer
        
        h0 - initial output (float)
        c0 - initial cell state (float)
        sigma_g - remembering, aquiring and output candidate activation 
                  functions (R->R)
        sigma_c - cell state activation function (R->R)
        sigma_h - output activation function (R->R)
        seed - seed for random fills of W,U,b parameters arrays
        """
        
        self.h0 = h0
        self.c0 = c0
        self.sigma_g = sigma_g
        self.sigma_c = sigma_c
        self.sigma_h = sigma_h
        self.seed = seed
        
        if self.seed is not None:
            np.random.seed(self.seed)

        self.W = np.random.rand(4)
        self.U = np.random.rand(4)
        self.b = np.random.rand(4)
     
    def _fiosch_call(self,xs,W,U,b) : 
        """
        Compute an exploded view of a call to the network
        
        Input
        xs - 1d array of inputs
        W - 1x4 array input parameter matrix
        U - 1x4 array memory parameter matrix
        b - 1x4 array bias parameter matrix
        
        Output
        f - forget gate activation
        i - input gate activation
        o - output gate activation
        s - cell state activation
        c - cell state
        h - output
        """
        
        n = len(xs)
        
        h = np.zeros(n)
        c = np.zeros(n)
        
        f = np.zeros(n)
        i = np.zeros(n)
        o = np.zeros(n)
        s = np.zeros(n)
        
        h_tm1 = self.h0
        c_tm1 = self.c0
        
        for j in range(n) : 
            
            val = xs[j]*W + h_tm1*U + b
        
            f[j] = self.sigma_g(np.array([val[0]]))[0]
            i[j] = self.sigma_g(np.array([val[1]]))[0]
            o[j] = self.sigma_g(np.array([val[2]]))[0]
            s[j] = self.sigma_c(np.array([val[3]]))[0]
            
            c[j] = f[j]*c_tm1 + i[j]*s[j]
            h[j] = o[j]*self.sigma_h(np.array([c[j]]))[0]
            
            h_tm1 = h[j]
            c_tm1 = c[j]
        
        return [f,i,o,s,c,h]
        
    def __call__(self,xs) :
        """
        call the neuron with an array of inputs
        
        Input
        xs - 1d array of inputs
        
        Output
        h - 1d array of output
        """
        
        return self._fiosch_call(xs,self.W,self.U,self.b)[-1]

    def numerical_derivative(self,w,dw,xs,ys,WUb_flag,coord) : 
        """
        Numerically compute the derivative of the sum of squares loss function
        using finite differences
        
        Input
        w - the value of the weight at which to compute the derivative
        dw - the size of the finite difference step
        xs - 1d array of sample inputs
        ys - 1d array of sample outputs
        WUb_flag - string in ['W', 'U', 'b'] indicating which parameter group
                   with respect to which the derivative should be computed
        coord - index into the parameter group array
        
        Output
        Numerical derivative at w of requested coordinate of the loss function
        """
         
        def funct(ww) : 
            
            W = np.array(self.W)
            U = np.array(self.U)
            b = np.array(self.b)
            
            if WUb_flag == 'W' : 
                W[coord] = ww
            elif WUb_flag == 'U' :
                U[coord] = ww 
            else : 
                b[coord] = ww

            f,i,o,s,c,h = self._fiosch_call(xs,W,U,b)

            return 0.5*np.sum((h - ys)**2)

        return n_der(funct,w,dw)
        
    def analytic_derivative(self,xs,ys) :
        """
        Compute the analytic derivative of the sum of squares loss functions
        with respect to the weights using the chain rule
        
        Input
        xs - 1d array of sample inputs
        ys - 1d array of sample outputs
        
        Output
        1x12 gradient matrix organized as 
            [W_f, U_f, b_f, W_i, U_i, b_i, W_o, U_o, f_o, W_c, U_c, f_c]
        """
        
        n = len(xs)
        
        h_tm1 = self.h0
        c_tm1 = self.c0
        dc_dw_tm1 = np.zeros(12)
        dh_dw_tm1 = np.zeros(12)
        
        result = np.zeros(12)
        
        for j in range(n) : 
            
            val = xs[j]*self.W + h_tm1*self.U + self.b

            f = self.sigma_g(np.array([val[0]]))[0]
            i = self.sigma_g(np.array([val[1]]))[0]
            o = self.sigma_g(np.array([val[2]]))[0]
            s = self.sigma_c(np.array([val[3]]))[0]
            
            c = f*c_tm1 + i*s
            h = o*self.sigma_h(np.array([c]))[0]
            
            df_dw = self.U[0]*dh_dw_tm1
            df_dw += np.array([xs[j], h_tm1, 1.] + 9*[0.])
            df_dw *= self.sigma_g.deriv_diag(np.array([val[0]]))[0]
                                                      
            di_dw = self.U[1]*dh_dw_tm1
            di_dw += np.array(3*[0.]+[xs[j], h_tm1, 1.]+6*[0.])
            di_dw *= self.sigma_g.deriv_diag(np.array([val[1]]))[0]
                                                      
            do_dw = self.U[2]*dh_dw_tm1
            do_dw += np.array(6*[0.]+[xs[j], h_tm1, 1.]+3*[0.])
            do_dw *= self.sigma_g.deriv_diag(np.array([val[2]]))[0]
                                             
            ds_dw = self.U[3]*dh_dw_tm1
            ds_dw += np.array(9*[0.]+[xs[j], h_tm1, 1.])
            ds_dw *= self.sigma_c.deriv_diag(np.array([val[3]]))[0]
                                         
            dc_dw = c_tm1*df_dw + f*dc_dw_tm1
            dc_dw += s*di_dw + i*ds_dw
                
            dh_dw = self.sigma_h(np.array([c]))[0]*do_dw
            dh_dw += o*self.sigma_h.deriv_diag(np.array([c]))[0]*dc_dw
                                                        
            result += (h - ys[j])*dh_dw
                                                        
            h_tm1 = h
            c_tm1 = c
        
            dh_dw_tm1 = dh_dw
            dc_dw_tm1 = dc_dw
            
        return result
            

class T_lstm : 
    """
    Traditional lstm networks
    
    see https://en.wikipedia.org/wiki/Long_short-term_memory
    and http://colah.github.io/posts/2015-08-Understanding-LSTMs/
    """
    
    def __init__(self,
                 n,
                 m,
                 h0,
                 c0,
                 sigma_g,
                 sigma_c,
                 sigma_h,
                 seed = None,
                 ) :
        """
        T_lstm initializer
        
        n - dimension of input
        m - dimension of output
        h0 - length m array
        c0 - length m array
        sigma_g - remembering, aquiring and output candidate activation 
                  functions (R^m->R^m)
        sigma_c - cell state activation function (R^m->R^m)
        sigma_h - output activation function (R^m->R^m)
        seed - seed for random fills of W,U,b parameters arrays
        """
        
        self.n = n
        self.m = m
        self.h0 = h0.copy()
        self.c0 = c0.copy()
        self.sigma_g = sigma_g
        self.sigma_c = sigma_c
        self.sigma_h = sigma_h
        self.seed = seed
                
        # the previous output
        self.h = np.zeros(self.m)
        
        # the previous cell state
        self.c = np.zeros(self.m)
        
        # seed the random generator if requested
        if self.seed is not None :
            np.random.seed(self.seed)
        
        # initial guesses for the weights
        self.W = np.random.rand(4*self.m,self.n)
        self.U = np.random.rand(4*self.m,self.m)
        self.b = np.random.rand(4*m)

    def _fiosch_call(self,xs,W,U,b) : 
        """
        Compute an exploded view of a call to the network
        
        Input
        xs - 1d array of inputs
        W - 1x4 array input parameter matrix
        U - 1x4 array memory parameter matrix
        b - 1x4 array bias parameter matrix
        
        Output
        f - forget gate activation
        i - input gate activation
        o - output gate activation
        s - cell state activation
        c - cell state
        h - output
        """
        
        n_smpls = len(xs)
        
        h = np.zeros((n_smpls,self.m))
        c = np.zeros((n_smpls,self.m))
        
        f = np.zeros((n_smpls,self.m))
        i = np.zeros((n_smpls,self.m))
        o = np.zeros((n_smpls,self.m))
        s = np.zeros((n_smpls,self.m))

        h_tm1 = self.h0
        c_tm1 = self.c0
        
        for j in range(n_smpls) : 
            
            val = W.dot(xs[j]) + U.dot(h_tm1) + b
                        
            f[j] = self.sigma_g(val[:self.m])
            i[j] = self.sigma_g(val[self.m:2*self.m])
            o[j] = self.sigma_g(val[2*self.m:3*self.m])
            s[j] = self.sigma_c(val[3*self.m:])
            
            c[j] = f[j]*c_tm1 + i[j]*s[j]
            h[j] = o[j]*self.sigma_h(c[j])
            
            h_tm1 = h[j]
            c_tm1 = c[j]

        return [f,i,o,s,c,h]
        
    def __call__(self,xs) :
        """
        call the neuron with an array of inputs
        
        Input
        xs - (? x self.n) array of inputs
        
        Output
        h - (? x self.m) array of output
        """
        
        return self._fiosch_call(xs,self.W,self.U,self.b)[-1]

    def numerical_derivative(self,w,dw,xs,ys,WUb_flag,coord) : 
        """
        Numerically compute the derivative of the sum of squares loss function
        using finite differences
        
        Input
        w - the value of the weight at which to compute the derivative
        dw - the size of the finite difference step
        xs - ? x self.n array of sample inputs
        ys - ? x self.m array of sample outputs
        WUb_flag - string in ['W', 'U', 'b'] indicating which parameter group
                   with respect to which the derivative should be computed
        coord - index into the parameter group array: should be 2 dimensional
                if WUb_flag is 'W' or 'U', and 1 dimensional if WUb_flag is 'b'
        
        Output
        Numerical derivative at w of requested coordinate of the loss function
        """
        
        # make sure coord has the right dimension
        
        if WUb_flag in ['W', 'U'] : 
            assert len(coord) == 2, "Coord should be length 2"
        elif WUb_flag == 'b' : 
            assert type(coord) == int, "Coord should be int"
        else : 
            raise ValueError("WUb_flag should be in ['W', 'U', 'b']")

        def funct(ww) : 
            
            W = np.array(self.W)
            U = np.array(self.U)
            b = np.array(self.b)
            
            if WUb_flag == 'W' : 
                W[coord] = ww
            elif WUb_flag == 'U' :
                U[coord] = ww 
            else : 
                b[coord] = ww

            f,i,o,s,c,h = self._fiosch_call(xs,W,U,b)

            v = h - ys
            
            return 0.5*np.sum(np.einsum('...i,...i',v,v))
   
        return n_der(funct,w,dw)
        
    def analytic_derivative(self,xs,ys) :
        """
        Compute the analytic derivative of the sum of squares loss functions
        with respect to the weights using the chain rule
        
        Input
        xs - ? x self.n array of sample inputs
        ys - ? x self.m array of sample outputs
        
        Output
        1 x 4*(self.m*self.n + self.m*self.m + self.m) gradient matrix 
        organized as 
            [W_f, U_f, b_f, W_i, U_i, b_i, W_o, U_o, f_o, W_c, U_c, f_c]
        where if we are thinking of the W and U parameters as 2 dimensional 
        matrices then the above scheme referes to these matrices flattened 
        row first
        """
        
        n_smpls = len(xs)
        n_w = self.m*self.n+self.m*self.m+self.m
        
        h_tm1 = list(self.h0)
        c_tm1 = self.c0
        dc_dw_tm1 = np.zeros((self.m,4*n_w))
        dh_dw_tm1 = np.zeros((self.m,4*n_w))
        
        result = np.zeros(4*n_w)                                                    
        
        # compute the row and column indices for the full derivative
        # matrices of f, i, o and s
        W_i = np.array([np.array([self.n*[j] for j in range(self.m)]).flatten(),
                        np.arange(self.n*self.m)])
        U_i = np.array([np.array([self.m*[j] for j in range(self.m)]).flatten(),
                        np.arange(self.m*self.m) + self.n*self.m])
        b_i = np.array([[j,self.m*self.n+self.m*self.m+j] 
                                            for j in range(self.m)]).T
        
        # loop over the samples to collect the derivative with 
        # the chain rule
        for j in range(n_smpls) : 
            
            x = list(xs[j])
            
            val = self.W.dot(x) + self.U.dot(h_tm1) + self.b
            
            f = self.sigma_g(val[:self.m])
            i = self.sigma_g(val[self.m:2*self.m])
            o = self.sigma_g(val[2*self.m:3*self.m])
            s = self.sigma_c(val[3*self.m:])
            
            c = f*c_tm1 + i*s
            h = o*self.sigma_h(c)
            
            df_dw = self.U[:self.m].dot(dh_dw_tm1)
            df_dw[W_i[0],W_i[1]] += self.m*x
            df_dw[U_i[0],U_i[1]] += self.m*h_tm1
            df_dw[b_i[0],b_i[1]] += 1.
            df_dw = self.sigma_g.deriv_diag(val[:self.m])[:,np.newaxis]*df_dw
                                            
            di_dw = self.U[self.m:2*self.m].dot(dh_dw_tm1)
            di_dw[W_i[0],W_i[1]+n_w] += self.m*x
            di_dw[U_i[0],U_i[1]+n_w] += self.m*h_tm1
            di_dw[b_i[0],b_i[1]+n_w] += 1.
            di_dw = self.sigma_g.deriv_diag(val[self.m:2*self.m])[:,np.newaxis]*di_dw
                                            
            do_dw = self.U[2*self.m:3*self.m].dot(dh_dw_tm1)
            do_dw[W_i[0],W_i[1]+2*n_w] += self.m*x
            do_dw[U_i[0],U_i[1]+2*n_w] += self.m*h_tm1
            do_dw[b_i[0],b_i[1]+2*n_w] += 1.
            do_dw = self.sigma_g.deriv_diag(val[2*self.m:3*self.m])[:,np.newaxis]*do_dw
                                
            ds_dw = self.U[3*self.m:].dot(dh_dw_tm1)
            ds_dw[W_i[0],W_i[1]+3*n_w] += self.m*x
            ds_dw[U_i[0],U_i[1]+3*n_w] += self.m*h_tm1
            ds_dw[b_i[0],b_i[1]+3*n_w] += 1.
            ds_dw = self.sigma_c.deriv_diag(val[3*self.m:])[:,np.newaxis]*ds_dw
                                            
            dc_dw = c_tm1[:,np.newaxis]*df_dw + f[:,np.newaxis]*dc_dw_tm1
            dc_dw += s[:,np.newaxis]*di_dw + i[:,np.newaxis]*ds_dw

            dh_dw = self.sigma_h(c)[:,np.newaxis]*do_dw
            dh_dw += o[:,np.newaxis]*self.sigma_h.deriv_diag(c)[:,np.newaxis]*dc_dw

            result += (h - ys[j]).dot(dh_dw)
                                                        
            h_tm1 = list(h)
            c_tm1 = c
        
            dh_dw_tm1[:] = dh_dw
            dc_dw_tm1[:] = dc_dw
            
        return result

