#Implements all layers in a CNN
#Layers could remember the inputs themselves instead of model.py maintaining the input for them
import numpy as np
from im2col import im2col_indices, col2im_indices

class layer:
	def __init__(self):
		self._input = None
		self._output = None

	def set_input(self, _input):
		self._input = _input

	def get_input(self):
		return self._input

	def set_output(self, _output):
		self._output = _output

	def get_output(self):
		return self._output

class linear_layer(layer):
	def __init__(self, input_D, output_D):
		self.weights = np.random.normal(0, 0.1, (input_D, output_D))
		self.biases = np.random.normal(0, 0.1, (1, output_D))

		self.dw = np.zeros((input_D, output_D))
		self.db = np.zeros((1, output_D))
	
	def forward(self, X):
		forward_output = np.dot(X, self.weights) + self.biases
		#print forward_output.shape, X.shape, self.weights.shape, self.biases.shape
		return forward_output

	def backward(self, X, grad):
		#print X.shape, grad.shape
		#self.dw += np.dot(X.T, grad)
		self.dw += np.outer(X, grad)
		#self.db += np.sum(grad, axis=0)
		self.db += grad
		backward_output = np.dot(grad, self.weights.T)

		return backward_output

	def update_weights(self, learning_rate):
		#print "Updating weights"
		self.weights -= self.dw * learning_rate
		self.biases -= self.db * learning_rate
		#print self.weights[:10]
		#print self.biases
		#TODO: Not required when batching support is added
		self.dw = np.zeros(self.weights.shape)
		self.db = np.zeros(self.biases.shape)

class convolution_layer(layer):

	def __init__(self, field, padding, stride, depth, filters):
		epsilon = 0.12

		self.field = field
		self.padding = padding
		self.stride = stride
		self.depth = depth
		self.filters = filters
	    
		self.weights = np.random.rand(filters, field * field * depth) * 2 * epsilon - epsilon
		self.biases = np.random.rand(filters)
		self.dw = np.zeros((self.weights.shape))
		self.db = np.zeros((self.biases.shape))

	"""
	D_out = filters
	W_in, H_in, D_in = X.shape
	W_out = (W_in - F + 2P)/S + 1
	weights : [D_out X FFD_in]
	biases : [D_out X 1]
	"""
	def forward(self, X):
	    #Image size is W * H * D
	    W_in, _, _ = X.shape
	    W_out = int((W_in - self.field + 2 * self.padding)/self.stride) + 1
	    
	    #Preprocessing X for im2col library 
	    X_pre = np.expand_dims(np.rollaxis(X, 2), axis=0)
	    X_columnar = im2col_indices(X_pre, self.field, self.field, self.padding, self.stride) #[FFD_in X W_out**2]
	    
	    result = np.dot(self.weights, X_columnar) # [ D_out X W_out**2 ]
	    result = np.reshape(result.T, (W_out, W_out, self.filters))
	    result += self.biases
	    return result

	"""
	a = activations in the current layer
	dd = gradients of the next layer
	dWeight = dd * a
	dBiase = dd
	dActivtion = dd * Weight

	D_out = filters
	W_out = (W_in - F + 2P)/S + 1
	"""
	def backward(self, X, grad):
	    
	    W_out, H_out, D_out = grad.shape
	    W_in, H_in, D_in = X.shape

	    #Preprocess 'a' for im2col utility
	    X = np.expand_dims(np.rollaxis(X, 2), axis=0)
	    a_columnar = im2col_indices(X, self.field, self.field, self.padding, self.stride)#[FFD X W_out*H_out]
	    grad = grad.reshape(W_out * H_out, D_out)
	    
	    dWeight = np.dot(a_columnar, grad).T #[ D_out X FFD ]
	    dBias = np.sum(grad, axis=0)
	    dActivation = np.dot(grad, self.weights).T #[FFD_out X W_out**2]
	    dActivation = col2im_indices (dActivation, (1, D_in, W_in, H_in),
	            self.field, self.field, self.padding, self.stride) # 1 X D_in X W_in X H_in
	    #Convert to required format
	    dActivation = np.squeeze(dActivation, axis=0)
	    #Move D axis to end
	    dX = np.rollaxis(dActivation, 0, 3) #W_in X H_in X D_in

	    self.dw += dWeight
	    self.db += dBias
	    return dX

	def update_weights(self, learning_rate):
		self.weights -= self.dw * learning_rate
		self.biases -= self.db * learning_rate
		self.dw = np.zeros((self.weights.shape))
		self.db = np.zeros((self.biases.shape))

class softmax_cross_entropy(layer):
	def forward(self, X):
		R = X.ravel()
		#print R
		R = R - np.max(R)
		R_exp = np.exp(R)
		R = R_exp / np.sum(R_exp)

		return R

	def backward(self, yhat, y):
		grad = yhat - y
		#print "Gradient: ", grad
		#TODO: return 3 dimensional
		#grad = grad[None, None, :]
		return grad

class relu_layer(layer):

	def forward(self, X):
		return np.maximum(0, X)

	def backward(self, X, grad):
    	#dd : W X H X D
    	#a  : W X H X D
		return (X > 0) * grad

#Taken from CSCI 567 Assignment 2
class flatten_layer(layer):

    def __init__(self):
        self.size = None

    def forward(self, X):
        self.size = X.shape
        out_forward = X.reshape(-1)

        return out_forward

    def backward(self, X, grad):
        out_backward = grad.reshape(self.size)

        return out_backward

#Taken from CSCI 567 Assignment 2
class dropout_layer(layer):
	def __init__(self, r):
		self.r = r
		self.mask = None

	def forward(self, X, is_train=False):
		if is_train:
			self.mask = (np.random.uniform(0.0, 1.0, X.shape) >= self.r).astype(float) * (1.0 / (1.0 - self.r))
		else:
			self.mask = np.ones(X.shape)
		return np.multiply(X, self.mask)

	def backward(self, X, grad):
		return self.mask * grad
