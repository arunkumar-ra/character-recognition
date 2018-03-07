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
import argparse

num_examples = 1000

def svhn_dataset():
    #http://ufldl.stanford.edu/housenumbers/train_32x32.mat
    mat = spio.loadmat('/Users/arun/Downloads//datasets/SVHN/train_32x32.mat')
    X = mat['X']
    y = (mat['y'] - 1).ravel()

    X = np.rollaxis(X, 3, 0)
    #visualize.show_svhn_dataset(X)
    print X.shape, y.shape
    return (X, y)

def clean_svhn_dataset(count=73257):
    #http://ufldl.stanford.edu/housenumbers/train_32x32.mat
    mat = spio.loadmat('/Users/arun/Downloads/datasets/SVHN/train_32x32.mat')
    X = mat['X']
    y = (mat['y'] - 1).ravel()

    greyX = np.mean(X, dtype=np.uint8, axis=2)
    X = np.reshape(greyX, (1024,73257)).T #Convert to 2D.
    print X.shape, y.shape
    #visualize.show_svhn_dataset(X)

    #X_train, y_train = train_test_split(X, y, train_size=count, random_state=26, stratify=y)
    return (X[:count], y[:count])

def nist_dataset():
    mat = spio.loadmat('mnist.mat')
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

def main(X, y):
    print "Loading data..."

    #Preprocess data by shuffling and splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)

    model = cnn.train(X_train, y_train)
    probs = cnn.predict(model, X_test)
    
    predictions = np.argmax(probs, axis=1)
    accuracy = sum(predictions == y_test) / len(y_test)

    print "Test set accuracy = ", accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick nist/svhn to process.")
    parser.add_argument("dataset", help="nist/svhn")
    args = parser.parse_args()

    np.random.seed(0)
    X, y = (None, None)
    
    if args.dataset == "svhn":
        X, y = svhn_dataset()
    elif args.dataset == "nist":
        X, y = nist_dataset()

    main(X, y)
