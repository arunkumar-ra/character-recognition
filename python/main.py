#!/usr/bin/python

from __future__ import division
from sklearn.model_selection import train_test_split
import sklearn.datasets
import scipy.io as spio
import numpy as np

import nnet
import visualize
import predict
import activation
import cnn

num_examples = 73257

num_layers = 3
#layer_size = [2, 5, 2]
#layer_size = [1024, 500, 10]

def svhn_dataset():
    mat = spio.loadmat('/Users/arun/Downloads/datasets/SVHN/train_32x32.mat')
    X = mat['X']
    y = (mat['y'] - 1).ravel()

    X = np.rollaxis(X, 3, 0)
    #visualize.show_svhn_dataset(X)

    return (X, y)

def clean_svhn_dataset(count=73257):
    mat = spio.loadmat('/Users/arun/Downloads/datasets/SVHN/train_32x32.mat')
    X = mat['X']
    y = (mat['y'] - 1).ravel()

    greyX = np.mean(X, dtype=np.uint8, axis=2)
    X = np.reshape(greyX, (1024,73257)).T #Convert to 2D.
    print X.shape, y.shape
    visualize.show_svhn_dataset(X)

    #X_train, y_train = train_test_split(X, y, train_size=count, random_state=26, stratify=y)
    return (X[:count], y[:count])

def nist_dataset():
    mat = spio.loadmat('/Users/arun/Downloads/datasets/MNIST/data.mat')
    X = mat['X']
    y = (mat['y'] % 10).ravel()

    m = len(X)

    X = np.reshape(X, (m, 20, 20, 1))
    return X, y

def cnn_sklearn_digits():
    X, y = sklearn.datasets.load_digits(return_X_y=True)
    X = np.reshape(X, (1797, 8, 8, 1))
    return X, y

def run():

    print "Loading data..."
    np.random.seed(0)
    X, y = sklearn.datasets.make_moons(n_samples=num_examples, noise=0.20)
    #TODO: add feature scalig for this to work
    #X, y = sklearn.datasets.load_wine(True)
    #X, y = sklearn.datasets.load_digits(return_X_y=True)
    #X, y = svhn_dataset(num_examples)

    X_tc, X_test, y_tc, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tc, y_tc, test_size=0.2, random_state=26, stratify=y_tc)

    #visualize.scatter_plot(X_train, y_train)
    
    #TODO: ONLY single option should be set
    act_func = {"func": activation.tanh, "gradient": activation.tanh_gradient}
    
    hyperparams = nnet.fit_hyperparameters(X_train, y_train, X_val, y_val, act_func, layer_size)
    print "Best parameters : ", hyperparams

    model = nnet.build_model(X_tc, y_tc, act_func, hyperparams, layer_size)
    predictions = predict.predict(model, X_test, act_func)
    
    accuracy = sum(predictions == y_test) / len(y_test)
    print predictions
    print y_test

    print "Test set Accuracy = ", accuracy
 
    #visualize.plot_decision_boundary(model, X, y, act_func)

def svhn():
    print "Loading data..."
    np.random.seed(0)

    #X, y = svhn_dataset()
    #X, y = cnn_sklearn_digits()
    X, y = nist_dataset()

    #Preprocess data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)

    #TODO: Fit hyperparameters

    F1, P1, S1, K1 = 8, 0, 1, 40
    F2, P2, S2, K2 = 13, 0, 1, 10
    image_depth = 1

    arc = ({'level': 1, 'layer': 'conv', 'weights': getWeights(F1, F1, image_depth, K1), 'biases': getBiases(K1), 'params' : (F1, P1, S1, K1) },
            {'level': 2, 'layer': 'relu'},
            {'level': 3, 'layer': 'conv', 'weights': getWeights(F2, F2, K1, K2), 'biases': getBiases(K2), 'params': (F2, P2, S2, K2) },
        )

    """
    
    arc = (
        {'level': 1, 'layer': 'conv', 'weights': getWeights(8, 8, 1, 20), 'biases': getBiases(20), 'params': (8, 0, 1, 20) },
        {'level': 2, 'layer': 'relu'},
        {'level': 3, 'layer': 'conv', 'weights': getWeights(1, 1, 20, 10), 'biases': getBiases(10), 'params': (1, 0, 1, 10) }
            )
    """
    cnn.train(X_train, y_train, arc)
    probs = cnn.predict(arc, X_test)
    
    predictions = np.argmax(probs, axis=1)

    print predictions
    print y_test

    accuracy = sum(predictions == y_test) / len(y_test)

    print "Test set accuracy = ", accuracy

def getWeights(W, H, D, K):
    epsilon = 0.012
    return np.random.rand(K, W*H*D) * 2 * epsilon - epsilon

def getBiases(K):
    return np.random.rand(K)

def test():
    X = np.array([[1, 2, 3], [4, 5, 4], [3, 2, 1]])
    X = X[None,:,:,None]
    y = [0]

    W1 = np.array([[1, 2, 3, 4]])
    W2 = np.array([[1, 0, 0, -1], [-1, 0, 0, -1]])
    b1 = np.array([0])
    b2 = np.array([1, 1])

    F1, P1, S1, K1 = 2, 0, 1, 1
    F2, P2, S2, K2 = 2, 0, 1, 2

    arc = (
            {'level': 1, 'layer': 'conv', 'weights': W1, 'biases': b1, 
                'params': (F1, P1, S1, K1)
                },
            {'level': 2, 'layer': 'conv', 'weights': W2, 'biases': b2,
                'params': (F2, P2, S2, K2)
                },
        )

    r, a = cnn.forward(X, 0, arc)
    R = np.array([1, 0])
    Y = np.array([0, 1])

    dd = R - Y
    dd = dd[None, None, :]

    cnn.backward(dd, a, arc, 1)

