#!/usr/bin/python

from __future__ import division
import numpy as np 
import sklearn 
import sklearn.linear_model 
import math
from scipy.optimize import minimize

import activation
import predict

def cost_gradient_function(nn_params, X, y, lam, layer_size, act_func):

    l1 = layer_size[0]
    l2 = layer_size[1]
    l3 = layer_size[2]
    
    #TODO: make simpler parsing method
    W1 = nn_params[0:l1*l2].reshape(l1, l2)
    W2 = nn_params[l1*l2:l1*l2 + l2*l3].reshape(l2, l3)
    b1 = nn_params[l1*l2 + l2*l3: l1*l2 + l2*l3 + l2]
    b2 = nn_params[l1*l2 + l2*l3 + l2: l1*l2 + l2*l3 + l2 + l3]

    m = len(X)

    W1_gradient = np.zeros((l1, l2))
    W2_gradient = np.zeros((l2, l3))
    b1_gradient = np.zeros(l2)
    b2_gradient = np.zeros(l3)

    J = 0
    
    activation_function = act_func['func']
    activation_gradient = act_func['gradient']

    #Forward Propagation
    z2 = np.matmul(X, W1) + b1
    a2 = activation_function(z2)
    z3 = np.matmul(a2, W2) + b2
    z3_exp = np.exp(z3)
    a3 = z3_exp / np.sum(z3_exp, axis=1)[:,None] #Softmax : [m * l3]

    #Backward Propagation
    Y = np.zeros((m, l3))
    Y[np.arange(len(y)), y] = 1

    J += (-1/m) * np.sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3))
   
    d3 = a3 - Y
    d2 = np.matmul(d3, W2.T) * activation_gradient(a2)

    W2_gradient += np.matmul(a2.T, d3)
    b2_gradient += np.sum(d3, axis=0)
    W1_gradient += np.matmul(X.T, d2)
    b1_gradient += np.sum(d2, axis=0)

    #Adding regularization
    J += lam/(2*m) * (np.sum(W2*W2) + np.sum(W1*W1))
    W1_gradient = (1/m) * (W1_gradient + lam * W1)
    W2_gradient = (1/m) * (W2_gradient + lam * W2)
    b1_gradient = 1/m * b1_gradient
    b2_gradient = 1/m * b2_gradient

    print "Cost : ", J
    nn_gradients = np.concatenate((W1_gradient.ravel(), W2_gradient.ravel(), b1_gradient.ravel(), b2_gradient.ravel()))
    return (J, nn_gradients)

def build_model(X, y, act_func, hyperparams, layer_size):
    l1 = layer_size[0]
    l2 = layer_size[1]
    l3 = layer_size[2]
    
    W1 = np.random.rand(l1, l2)
    W2 = np.random.rand(l2, l3)
    b1 = np.random.rand(l2)
    b2 = np.random.rand(l3)

    m = len(X)

    lam = hyperparams['lam']

    nn_params = np.concatenate((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))
    
    minimized = minimize(cost_gradient_function, nn_params, args=(X, y, lam, layer_size, act_func), method='CG', jac=True, options = {'disp': True, 'maxiter': 5000})
    print minimized
    
    nn_params = minimized.x
    W1 = nn_params[0:l1*l2].reshape(l1, l2)
    W2 = nn_params[l1*l2:l1*l2 + l2*l3].reshape(l2, l3)
    b1 = nn_params[l1*l2 + l2*l3: l1*l2 + l2*l3 + l2]
    b2 = nn_params[l1*l2 + l2*l3 + l2: l1*l2 + l2*l3 + l2 + l3]
       
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def fit_hyperparameters(X_train, y_train, X_val, y_val, act_func, layer_size):

    return {'lam': 0.1}
    lam = [0.1, 0.3, 1]
    alpha = [0.1, 0.3, 1]
    
    print "Fitting hyperparameters..."

    best_accuracy = 0
    for i in range (len(lam)):
        hyperparams = {'lam': lam[i]}
        model = build_model(X_train, y_train, act_func, hyperparams, layer_size)
        predictions = predict.predict(model, X_val, act_func)

        accuracy = sum(predictions == y_val) / len(y_val)
        print "Accuracy obtained : ", accuracy

        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_lam = lam[i]

    return {'lam': best_lam}
