from __future__ import division
import numpy as np
import layers
import model
from sklearn.model_selection import train_test_split

def get_model():
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

def get_batch(X, y, batch_size):
    N = len(X)
    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        yield (X_batch, y_batch)

def get_X_batch(X, batch_size):
    N = len(X)
    for i in range(0, N, batch_size):
        X_batch = X[i:i+batch_size]

        yield X_batch

def train(X, y, Model=get_model()):
    batch_size = Model.params['batch_size']
    learning_rate = Model.params['learning_rate']
    epochs = Model.params['epochs']
    num_classes = Model.params['num_classes']
    dropout_rate = Model.params['dropout_rate'] #Not required

    correct_predictions = np.zeros(10)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y) #0.25 is 1/5th of original dataset size
    X, y = (X_train, y_train)
    for i in range(epochs):
        J = 0
        correct_predictions = 0

        for X_batch, y_batch in get_batch(X, y, batch_size):
            N = len(y_batch)
            Y = np.zeros((N, num_classes)) #We use N = len(y_batch) because the last set might have fewer than batch_size examples
            Y[np.arange(N), y_batch] = 1

            yhat = Model.forward(X_batch, is_train=True)
            Model.backward(yhat, Y)

            predictions = np.argmax(yhat, axis=1)
            #TODO: Count each number separately
            correct_predictions += sum(predictions == y_batch)
            
            #TODO: how to avoid log of 0
            #J -= np.sum(Y * np.log(yhat) + (1 - Y) * np.log(1 - yhat))

            Model.update_weights(learning_rate)

            # if (j+1) % 1000 == 0:
            #     print "Cost in ", j+1, " iterations = ", J/(j+1)
            #     print "Correct predictions count ", correct_predictions
            #     print "Total correct ", np.sum(correct_predictions), "/", (j+1)

        #Checking against cross validation set to prevent overfitting
        cross_val_prediction_probabilities = predict(Model, X_val, batch_size)
        cross_val_predictions = np.argmax(cross_val_prediction_probabilities, axis=1)
        prediction_accuracy = sum(cross_val_predictions == y_val) / len(y_val)

        print "Epoch: ", i+1
        print "Training prediction accuracy: %.3f" % (correct_predictions/len(X))
        print "Cross validation accuracy: ", prediction_accuracy

    return Model

def predict(Model, X_test, batch_size=20):
    predictions = []
    for X_batch in get_X_batch(X_test, batch_size):
        probs = Model.forward(X_batch, is_train = False)
        
        predictions.append(probs)

    predictions = np.array(predictions)

    return np.reshape(predictions, (len(X_test), predictions.shape[2]))