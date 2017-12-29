#!/usr/bin/python

from __future__ import division
import numpy as np 
import sklearn 
import sklearn.datasets 
import sklearn.linear_model 
import math
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize

import activation
import predict
import visualize

input_layer_size = 64
hidden_layer_size = 50
num_labels = 10
num_examples = 2000

def randInitialize(L_in, L_out):
    return np.random.rand(L_out, L_in)
    # TODO: Normalze over epsilon

def cost_gradient_function(nn_params, X, y, lam, l1, l2, l3, act_func):
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

    #TODO: Vectorize
    #Calculate error 
    for i in range (m):
        #Forward Propagation
        z2 = np.matmul(X[i], W1) + b1
        a2 = activation_function(z2)
        z3 = np.matmul(a2, W2) + b2
        z3_exp = np.exp(z3)
        a3 = z3_exp / sum(z3_exp) #Softmax

        #Backward Propagation
        Y = np.zeros(num_labels)
        Y[y[i]] = 1
        J += (-1/m) * sum(Y * np.log(a3) + (1 - Y) * np.log(1 - a3)) 
       
        d3 = a3 - Y
        d2 = np.matmul(d3, W2.T) * activation_gradient(a2)
    
        W2_gradient += np.outer(d3, a2).T
        b2_gradient += d3 
        W1_gradient += np.outer(d2, X[i]).T
        b1_gradient += d2

    #Adding regularization
    J += lam/(2*m) * (sum(sum(W2*W2)) + sum(sum(W1*W1)))
    W1_gradient = (1/m) * (W1_gradient + lam * W1)
    W2_gradient = (1/m) * (W2_gradient + lam * W2)
    b1_gradient = 1/m * b1_gradient
    b2_gradient = 1/m * b2_gradient

    print "Cost : ", J
    nn_gradients = np.concatenate((W1_gradient.ravel(), W2_gradient.ravel(), b1_gradient.ravel(), b2_gradient.ravel()))
    return (J, nn_gradients)

def build_model(X, y, act_func, hyperparams):
    W1 = np.random.rand(input_layer_size, hidden_layer_size)
    W2 = np.random.rand(hidden_layer_size, num_labels)
    b1 = np.random.rand(hidden_layer_size)
    b2 = np.random.rand(num_labels)

    m = len(X)

    lam = hyperparams['lam']

    nn_params = np.concatenate((W1.ravel(), W2.ravel(), b1.ravel(), b2.ravel()))
    
    #TODO: use minimization function
    minimized = minimize(cost_gradient_function, nn_params, args=(X, y, lam, input_layer_size, hidden_layer_size, num_labels, act_func), method='BFGS', jac=True, options = {'disp': True, 'maxiter': 100})
    print minimized
    
    l1 = input_layer_size
    l2 = hidden_layer_size
    l3 = num_labels

    nn_params = minimized.x

    W1 = nn_params[0:l1*l2].reshape(l1, l2)
    W2 = nn_params[l1*l2:l1*l2 + l2*l3].reshape(l2, l3)
    b1 = nn_params[l1*l2 + l2*l3: l1*l2 + l2*l3 + l2]
    b2 = nn_params[l1*l2 + l2*l3 + l2: l1*l2 + l2*l3 + l2 + l3]
       
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def fit_hyperparameters(X_train, y_train, X_val, y_val, act_func):

    #return {'lam': 0.1}
    lam = [0.1, 0.3, 1]
    alpha = [0.1, 0.3, 1]
    
    print "Fitting hyperparameters..."

    best_accuracy = 0
    for i in range (len(lam)):
        hyperparams = {'lam': lam[i]}
        model = build_model(X_train, y_train, act_func, hyperparams)
        predictions = predict.predict(model, X_val, act_func)

        accuracy = sum(predictions == y_val) / len(y_val)
        print "Accuracy obtained : ", accuracy

        if (accuracy > best_accuracy):
            best_accuracy = accuracy
            best_lam = lam[i]

    return {'lam': best_lam}

def run():

    print "Loading data..."
    np.random.seed(0)
    #X, y = sklearn.datasets.make_moons(n_samples=num_examples, noise=0.20)
    #X, y = sklearn.datasets.make_classification(n_samples = num_examples, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1)
    #X, y = sklearn.datasets.make_blobs(n_samples=num_examples, n_features=2, centers=num_labels)
    #TODO: add feature scalig for this to work
    #X, y = sklearn.datasets.load_wine(True)
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    
    X_tc, X_test, y_tc, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tc, y_tc, test_size=0.2, random_state=26, stratify=y_tc)

    #visualize.scatter_plot(X_train, y_train)
    
    #TODO: ONLY single option should be set
    act_func = {"func": activation.tanh, "gradient": activation.tanh_gradient}
    
    hyperparams = fit_hyperparameters(X_train, y_train, X_val, y_val, act_func)
    print "Best parameters : ", hyperparams

    model = build_model(X_tc, y_tc, act_func, hyperparams)
    predictions = predict.predict(model, X_test, act_func)
    
    accuracy = sum(predictions == y_test) / len(y_test)
    print predictions
    print y_test

    print "Test set Accuracy = ", accuracy
    print "Woo hoo!"
 
    #visualize.plot_decision_boundary(model, X, y, act_func)
