import cnn
from nist import nist
import numpy as np

def canvas():
	return np.zeros((20, 20, 1))

def main():
    np.random.seed(0)
    X, y = nist.dataset()
    epsilon = 1e-4
    learning_rate = 0.01

    #Train with NIST dataset
    Model = cnn.train(X, y, nist.get_model())

    for i in range(10):
    	X = canvas()
    	Y = np.zeros(10)
    	Y[i] = 1
    	predicted_prob = 0

    	while 1 - predicted_prob > epsilon:
            yhat = Model.forward(X, is_train=false)
            predicted_prob = yhat[i]

            dx = Model.backward() ##UGH! Need to vectorize code before doing this. Otherwise weights keep accumulating
            X -= dx * 0.01
            print X
    	#Visualize X



if __name__ == "__main__":
    main()