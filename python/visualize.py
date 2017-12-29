import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import predict

def scatter_plot(X, y):
    #X has 2 features
    #y has integer values
    plt.scatter(X[:,0], X[:,1], c=y, s=40)
    plt.show()

def plot_decision_boundary(model, X, y, act_func):
    x_min = X[:,0].min() - 0.5
    x_max = X[:,0].max() + 0.5
    y_min = X[:,1].min() - 0.5
    y_max = X[:,1].max() + 0.5

    xx = np.linspace(x_min, x_max, 50)
    yy = np.linspace(y_min, y_max, 50)

    xg, yg = np.meshgrid(xx, yy)
    points = np.transpose([xg.ravel(), yg.ravel()])

    predictions = predict.predict(model, points, act_func)
    
    Z = predictions.reshape(xg.shape)
    plt.contourf(xg, yg, Z) 
    plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap=plt.cm.Spectral)
    plt.show()
