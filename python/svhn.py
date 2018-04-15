import scipy.io as spio
import numpy as np

class svhn:

    @classmethod
    def dataset(self):
        #http://ufldl.stanford.edu/housenumbers/train_32x32.mat
        mat = spio.loadmat('/Users/arun/Downloads//datasets/SVHN/train_32x32.mat')
        X = mat['X']
        y = (mat['y'] - 1).ravel()

        X = np.rollaxis(X, 3, 0)
        #visualize.show_svhn_dataset(X)
        print X.shape, y.shape
        return (X, y)

    @classmethod
    def clean_dataset(self, count=73257):
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
