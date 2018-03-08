class model:
	def __init__(self):
		self.layers = []
		self.forward_output = None
		self.backward_output = None
		self.loss_layer = None
		self.params = None

	def add(self, layer):
		self.layers.append(layer)

	def set_loss(self, layer):
		self.layers.append(layer)
		self.loss_layer = layer

	def set_hyper_params(self, params):
		self.params = params

	def forward(self, X, is_train = False):
		if self.loss_layer == None:
			print "Loss function not defined."

		next_input = X
		for layer in self.layers:
			layer.set_input(next_input)
			if layer.__class__.__name__ == "dropout_layer":
				next_input = layer.forward(next_input, is_train)
			else:
				next_input = layer.forward(next_input)

		return next_input

	def backward(self, yhat, y):
		self.backward_output = []
		
		gradient = self.layers[-1].backward(yhat, y) #loss layer
		for layer in reversed((self.layers[:-1])):
			original_input = layer.get_input()

			gradient = layer.backward(original_input, gradient)

		return gradient

	def update_weights(self, learning_rate):
		for layer in self.layers:
			update_weights = getattr(layer, 'update_weights', None)
			if callable(update_weights):
				update_weights(learning_rate)
