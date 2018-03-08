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
import nist
import svhn

num_examples = 1000

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
    #X, y = svhn.dataset(num_examples)

    X_tc, X_test, y_tc, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_tc, y_tc, test_size=0.2, random_state=26, stratify=y_tc)

    #visualize.scatter_plot(X_train, y_train)
    
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
    
    np.random.seed(0)
    #Preprocess data by shuffling and splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=26, stratify=y)

    model = cnn.train(X_train, y_train, nist.get_model())
    probs = cnn.predict(model, X_test)
    
    predictions = np.argmax(probs, axis=1)
    accuracy = sum(predictions == y_test) / len(y_test)

    print "Test set accuracy = ", accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick nist/svhn to process.")
    parser.add_argument("dataset", help="nist/svhn")
    args = parser.parse_args()

    X, y = (None, None)
    
    if args.dataset == "svhn":
        X, y = svhn.dataset()
    elif args.dataset == "nist":
        X, y = nist.dataset()

    main(X, y)
