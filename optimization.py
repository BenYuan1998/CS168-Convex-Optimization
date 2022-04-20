#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 18:59:37 2022

@author: nuoyuan
"""

# %% Import all the necessary Python libraries
import numpy as np
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import pinv

# %% Define BackTrackingLineSearch, GradientDescentWithBackTracking, NewtonMethodWithBackTracking

def BackTrackingLineSearch(func, current_x, current_gradient, delta_x, alpha = 0.1, beta = 0.5):
    """
    Parameters:
    -----------
    func: the objective function
    current_x: the current value of the optimal point
    current_gradient: the gradient of the objective function at the current value of the optimal point (note: this needs to be a column vector)
    delta_x: search direction (note: this needs to be a column vector)
    alpha: hyperparamter,
        typical region is [0.01, 0.3]; the default value is 0.1
    beta: hyperparameter,
        typical region is [0.1, 0.8]; the default value is 0.5
    
    Returns:
    --------
    t: step size
    """
    t = 1 # initialize step size
    #print(func(current_x + t * delta_x))
    #print(func(current_x) + alpha * t * np.dot(current_gradient.T, delta_x))
    while func(current_x + t * delta_x) > func(current_x) + alpha * t * np.dot(current_gradient.T, delta_x): # guarantees that funct(x^{k+1}) <= funct(x^{k})
        t = beta * t # update t by backtracking
    return t
    
    
def GradientDescentWithBackTracking(func, ComputeGradient, init_x, epsilon = 1e-5):
    """
    Parameters
    ----------
    func : the objective function
    ComputeGradient : the function that computes the gradient of the objective function at the value of the optimal point corresponding to each iteration
    init_x : the initial guess about the optimal point
    epsilon : stopping criterion,
        When norm(gradient) < epsilon, the optimization algorithm terminates. The default is 1e-5.

    Returns
    -------
    optimal_point : the optimal point determined by the algorithm
    optimal_value : the corresponding optimal value of the objective function
    funcvals : a list that records the function values before convergence
    norms_gradient : a list that records the norm of the gradient at each iteration before convergence
    num_iter : a list that records the numbers of iteration needed for convergence
    """
    funcvals = list()
    num_iter = list()
    iteration = 0 # initialize iteration
    norms_gradient = list()
    norm_gradient = float("inf") # initialize the norm of the gradient
    current_x = init_x # initialze current_x
    while norm_gradient >= epsilon:
        current_gradient = ComputeGradient(current_x)
        norm_gradient = norm(current_gradient)
        #print(norm_gradient)
        norms_gradient.append(norm_gradient)
        delta_x = -current_gradient.reshape(-1, 1) # delta_x is set to be the negaive of the gradient
        t = BackTrackingLineSearch(func = func, current_x = current_x, current_gradient = current_gradient, delta_x = delta_x)
        current_x += t * delta_x # update current_x
        funcvals.append(func(current_x))
        iteration +=1
        num_iter.append(iteration)
    optimal_point = current_x
    optimal_value = func(optimal_point)
    return optimal_point, optimal_value, funcvals, norms_gradient, num_iter

def NewtonMethodWithBackTracking(func, ComputeGradient, ComputeHessian, init_x, epsilon = 1e-5):
    """
    Parameters
    ----------
    func : the objective function
    ComputeGradient : the function that computes gradient of the objective function at the value of the optimal point corresponding to each iteration
    ComputeHessian : the function that computes Hessian of the objective function at the value of the optimal point corresponding to each iteration
    init_x : the initial guess about the optimal point
    epsilon : stopping criterion,
        When Newton decrement < epsilon, the optimization algorithm terminates. The default is 1e-5.

    Returns
    -------
    optimal_point : the optimal point determined by the algorithm
    optimal_value : the corresponding optimal value of the objective function
    funcvals : a list that records the function values before convergence
    NetwonDecrements : a list that records the Newton decrement at each iteration before convergence
    num_iter : a list that records the numbers of iteration needed for convergence
    """
    funcvals = list()
    num_iter = list()
    iteration = 0 # initialize iteration
    NewtonDecrements = list()
    NewtonDecrement = float("inf") # initialize the Newton decrement
    current_x = init_x # initialize current_x
    while NewtonDecrement >= epsilon:
        current_gradient = ComputeGradient(current_x)
        current_hessian = ComputeHessian(current_x)
        try: inv_hessian = 1 / float(current_hessian) # if the objective function is univariate
        except: # if the objective function is multivariate
            try:
                inv_hessian = inv(current_hessian)
            except:
                inv_hessian = pinv(current_hessian)
        delta_x = -np.dot(inv_hessian, current_gradient).reshape(-1, 1)
        NewtonDecrement = np.dot(current_gradient.T, -delta_x)[0]
        #print(NewtonDecrement)
        NewtonDecrements.append(NewtonDecrement)
        t = BackTrackingLineSearch(func = func, current_x = current_x, current_gradient = current_gradient, delta_x = delta_x)
        current_x += t * delta_x # update current_x
        funcvals.append(func(current_x))
        iteration += 1
        num_iter.append(iteration)
    optimal_point = current_x
    opitmal_value = func(optimal_point)
    return optimal_point, opitmal_value, NewtonDecrements, num_iter

# a modified log function as a workaround for the DvisionbyZero error
def safe_ln(x):
    if x <= 0:
        return 0
    return np.log(x)




    
    
    
        
        
        
        
    