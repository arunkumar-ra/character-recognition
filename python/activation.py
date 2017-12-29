import numpy as np
import math

def sigmoid(z):
    sig = lambda t: 1/ (1 + math.exp(-t))
    vectorized_sigmoid = np.vectorize(sig)
    return vectorized_sigmoid(z)

def sigmoid_gradient(sz):
    return np.multiply(sz, 1-sz)

def tanh(z):
    return np.tanh(z)

def tanh_gradient(tz):
    return 1 - tz * tz
