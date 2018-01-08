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
    #print activations 
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
             
def convolution_forward(X, weights, biases, params):
    #Image size is W * H * D
    field, padding, stride, filters = params #F, P, S, K
    W, _, _ = X.shape

    #Preprocessing X to fit the library requirements
    X_pre = np.expand_dims(np.rollaxis(X, 2), axis=0)
    X_columnar = im2col_indices(X_pre, field, field, padding, stride) #[F * F * D x (W - F + 2P)/S + 1 squared]
    #weights : [filters x F * F * D]
    #biases : [filters x 1]

    result = np.dot(weights, X_columnar) # [ filters x (W - F + 2P)/S + 1 squared ]
    W2 = int((W - field + 2 * padding)/stride) + 1

    result = np.reshape(result.T, (W2, W2, filters))
    result += biases
    return result

def relu(X):
    return np.maximum(0, X)

def relu_gradient(a, dd):
    #dd : W X H X D
    #a  : W X H X D
    return (a > 0) * dd

def convolution_gradient(a, dd, weights, biases, params):
    field, padding, stride, filters = params
    """
    dWeight = dd *  a
    dBias   = dd 
    dActivation = dd * Weight

    D2 = filters
    W2 = (W - F + 2P) / S + 1
    """
    W2, H2, D2 = dd.shape
    W, H, D = a.shape

    dd = dd.reshape(W2 * H2, D2)
    #Preprocess a
    a = np.expand_dims(np.rollaxis(a, 2), axis=0) 
    a_columnar = im2col_indices(a, field, field, padding, stride) #[FFD X W2xH2]

    dWeight = np.dot(a_columnar, dd).T #[ D2 X FFD ]
    dBias = np.sum(dd, axis=0)
    dActivation = np.dot(dd, weights).T # FFD2 X W2**2
    dActivation = col2im_indices (dActivation, (1, D, W, H),
            field, field, padding, stride) # 1 X D X W X H
    #Convert to required format
    dActivation = np.squeeze(dActivation, axis=0)
    #Move D axis to end
    dActivation = np.rollaxis(dActivation, 0, 3) #W X H X D

    return (dWeight, dBias, dActivation)

def predict(arc, X_test):
    predictions = []
    for i in range (len(X_test)):
        result, activations = forward(X_test, i, arc)
        result_exp = np.exp(result)
        result = result_exp / np.sum(result_exp)
        result = result.ravel()
        #Now result stores the final list of probabilities
        
        #prediction = np.argmax(result)
        #predictions.append(prediction)
        predictions.append(result)

    return predictions

