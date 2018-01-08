import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image

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

def show_svhn_dataset(X):
    #Show first hundred examples
    X = X[:100]
    h = w = 32

    data = np.zeros((32*10, 32*10), dtype=np.uint8)
    for i in range (10):
        for j in range (10):
            k = 10 * i + j
            offset_x = i * w
            offset_y = j * h
            for x in range(w):
                for y in range(h):
                    data[offset_x + x, offset_y + y] = X[k][x * w + y]

    img = Image.fromarray(data)
    img.show()


