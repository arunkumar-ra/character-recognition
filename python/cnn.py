from __future__ import division
from im2col import im2col_indices, col2im_indices
import numpy as np

def train(X, y, arc):
    num_iter = 100
    for i in range(num_iter):
        print "Iteration # ", i + 1
        J, arc = iterate(X, y, arc)
        print "Cost : ", J, "\n"

    return arc

def iterate(X, y, arc, alpha = 0.001):
    m = len(X)
    correct_predictions = np.zeros(10)

    J = 0
    initializeGradients(arc)
    for i in range (len(X)):
        #Batch updates to gradient
        batch_size = m
        if (i + 1) % batch_size == 0:
            updateGradients(arc, alpha, batch_size)
        
        #Forward propagation
        result, activations = forward(X, i, arc)
        
        R = result.ravel()
        R_exp = np.exp(R)
        R = R_exp / np.sum(R_exp)
        #DEBUG
        if np.argmax(R) == y[i]:
            correct_predictions[y[i]] += 1

        #Cost
        final_layer = arc[-1]
        (_, _, _, num_classes) = final_layer['params']
        Y = np.zeros(num_classes)
        Y[y[i]] = 1

        J -= np.sum(Y * np.log(R) + (1 - Y) * np.log(1 - R))
        
        #Backward Propagation
        dd = R - Y
        dd = dd[None, None, :]
        backward(dd, activations, arc, alpha)

        if (i+1) % 1000 == 0:
            print "Cost in ", i , " iterations = ", J/(i+1)
            np.set_printoptions(precision=4)
            print "Correct predictions count ", correct_predictions
            print "Total correct ", np.sum(correct_predictions), "/", (i+1)
       
    return (J/m, arc)

def initializeGradients(arc):
    for j in range(len(arc)):
        layer = arc[j]
        if layer['layer'] == 'conv':
            layer['dw'] = np.zeros((layer['weights'].shape))
            layer['db'] = np.zeros((layer['biases'].shape))

def updateGradients(arc, alpha, batch_size):
    for j in range(len(arc)):
        layer = arc[j]
        if layer['layer'] == 'conv':
            layer['weights'] -= alpha * layer['dw'] / batch_size
            layer['biases'] -= alpha * layer['db'] / batch_size
    
    initializeGradients(arc)

def forward(X, i, arc):

    activations = (X[i], )
    result = X[i]
    for j in range (len(arc)):
        layer = arc[j]
        if layer['layer'] == 'conv':
            result = convolution_forward(result, layer['weights'],
                    layer['biases'], layer['params'])
        elif layer['layer'] == 'relu':
            result = relu(result)
        
        activations = activations + (result, )
    return result, activations

def backward(dd, activations, arc, alpha = 0.1):
    for j in range (len(arc) - 1, -1, -1):
        layer = arc[j]
        if layer['layer'] == 'conv':
            dw, db, dd = convolution_gradient(activations[j], dd,
                    layer['weights'], layer['biases'], layer['params'])
            layer['dw'] += dw
            layer['db'] += db
        if layer['layer'] == 'relu':
            dd = relu_gradient(activations[j], dd)
    pass

"""
W_in, H_in, D_in = X.shape
W_out = (W_in - F + 2P)/S + 1
weights : [D_out X FFD_in]
biases : [D_out X 1]
"""
def convolution_forward(X, weights, biases, params):
    #Image size is W * H * D
    field, padding, stride, D_out = params #F, P, S, K
    W_in, _, _ = X.shape
    W_out = int((W_in - field + 2 * padding)/stride) + 1
    
    #Preprocessing X for im2col library 
    X_pre = np.expand_dims(np.rollaxis(X, 2), axis=0)
    X_columnar = im2col_indices(X_pre, field, field, padding, stride) #[FFD_in X W_out**2]
    
    result = np.dot(weights, X_columnar) # [ D_out X W_out**2 ]
    result = np.reshape(result.T, (W_out, W_out, D_out))
    result += biases
    return result

def relu(X):
    return np.maximum(0, X)

def relu_gradient(a, dd):
    #dd : W X H X D
    #a  : W X H X D
    return (a > 0) * dd

"""
a = activations in the current layer
dd = gradients of the next layer
dWeight = dd * a
dBiase = dd
dActivtion = dd * Weight

D_out = filters
W_out = (W_in - F + 2P)/S + 1
"""
def convolution_gradient(a, dd, weights, biases, params):
    field, padding, stride, filters = params
    
    W_out, H_out, D_out = dd.shape
    W_in, H_in, D_in = a.shape

    #Preprocess 'a' for im2col utility
    a = np.expand_dims(np.rollaxis(a, 2), axis=0)
    a_columnar = im2col_indices(a, field, field, padding, stride)#[FFD X W_out*H_out]
    dd = dd.reshape(W_out * H_out, D_out)
    
    dWeight = np.dot(a_columnar, dd).T #[ D_out X FFD ]
    dBias = np.sum(dd, axis=0)
    dActivation = np.dot(dd, weights).T #[FFD_out X W_out**2]
    dActivation = col2im_indices (dActivation, (1, D_in, W_in, H_in),
            field, field, padding, stride) # 1 X D_in X W_in X H_in
    #Convert to required format
    dActivation = np.squeeze(dActivation, axis=0)
    #Move D axis to end
    dActivation = np.rollaxis(dActivation, 0, 3) #W_in X H_in X D_in

    return (dWeight, dBias, dActivation)

def predict(arc, X_test):
    predictions = []
    for i in range (len(X_test)):
        result, activations = forward(X_test, i, arc)
        result_exp = np.exp(result)
        result = result_exp / np.sum(result_exp)
        result = result.ravel() #Now result stores the final list of probabilities
        
        predictions.append(result)

    return predictions

