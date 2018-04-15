import scipy.io as spio
import numpy as np
import model
import layers

class nist:

	@classmethod
	def get_model(self):
		params = {
			'batch_size': 20,
			'learning_rate': 0.01,
			'epochs':  20,
			'num_classes':  10,
			'dropout_rate': 0.1
		}
		
		Model = model.model()
		Model.add(layers.convolution_layer(field=8, padding=0, stride=1, depth=1, filters=5))
		Model.add(layers.relu_layer())
		Model.add(layers.flatten_layer())
		Model.add(layers.dropout_layer(r = params['dropout_rate']))
		Model.add(layers.linear_layer(13*13*5, 10))
		Model.set_loss(layers.softmax_cross_entropy())
		Model.set_hyper_params(params)

		return Model

	@classmethod
	def get_nn_model(self):

		params = {
			'batch_size': 20,
			'learning_rate': 0.01,
			'epochs':  30,
			'num_classes':  10,
			'dropout_rate': 0.1
		}

		Model = model.model()
		Model.add(layers.flatten_layer())
		Model.add(layers.linear_layer(20*20, 200))
		Model.add(layers.dropout_layer(r = params['dropout_rate']))
		Model.add(layers.linear_layer(200, 10))
		Model.set_loss(layers.softmax_cross_entropy())
		Model.set_hyper_params(params)

		return Model

	@classmethod
	def dataset(self):
	    mat = spio.loadmat('mnist.mat')
	    X = mat['X']
	    y = (mat['y'] % 10).ravel()

	    m = len(X)

	    X = np.reshape(X, (m, 20, 20, 1))
	    return X, y
