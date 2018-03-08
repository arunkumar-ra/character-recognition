import numpy as np

#TODO: Predict probabilities
def predict(model, X, act_func):
    activation_function = act_func['func']

    z2 = np.matmul(X, model['W1']) + model['b1']
    a2 = activation_function(z2)
    z3 = np.matmul(a2, model['W2']) + model['b2']
    
    return np.argmax(z3, axis=1) #Max index in each row

