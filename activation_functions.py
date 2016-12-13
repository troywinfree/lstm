#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 19:26:01 2016

@author: troywinfree

various activation functions
"""

import numpy as np


class identity : 
    
    def __init__(self,n) : 
        """
        Initializer
        
        n - number of variables
        """
        
        self.n = n
        
    def __call__(self,x) : 
        """
        Evaluation method
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
     
        return x
        
    def deriv_diag(self,x) : 
        """
        Evaluate the derivative
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
            
        return np.ones(self.n)
        

class logistic : 
    
    def __init__(self,n) : 
        """
        Initializer
        
        n - number of variables
        """
        
        self.n = n
        
    def __call__(self,x) : 
        """
        Evaluation method
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
     
        return np.reciprocal(1. + np.exp(-x))
            
    def deriv_diag(self,x) : 
        """
        Evaluate the derivative
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
            
        exp_mx = np.exp(-x)
        
        return np.reciprocal((1.+exp_mx)**2)*exp_mx
        
        
class hyptan : 
    
    def __init__(self,n) : 
        """
        Initializer
        
        n - number of variables
        """
        
        self.n = n
        
    def __call__(self,x) : 
        """
        Evaluation method
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
     
        return np.tanh(x)
           
    def deriv_diag(self,x) : 
        """
        Evaluate the derivative
        
        x - array of length self.n
        """
        
        if len(x) != self.n : 
            raise ValueError
        
        return 1. - (np.tanh(x)**2)

