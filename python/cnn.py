from __future__ import division
import numpy as np
import layers
import model
from sklearn.model_selection import train_test_split

def train(X, y):
    batch_size = 20
    learning_rate = 0.01
    epochs = 20
    num_classes = 10
    dropout_rate = 0.1

    Model = model.model()
    Model.add(layers.convolution_layer(field=8, padding=0, stride=1, depth=1, filters=5))
    Model.add(layers.relu_layer())
    Model.add(layers.flatten_layer())
    Model.add(layers.dropout_layer(r = dropout_rate))
    Model.add(layers.linear_layer(13*13*5, 10))
    Model.set_loss(layers.softmax_cross_entropy())

    #Only flatten, linear and softmax would form a multi logistic regression 

    correct_predictions = np.zeros(10)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=27, stratify=y) #0.25 is 1/5th of original dataset size
    X, y = (X_train, y_train)
    for i in range(epochs):
        J = 0
        correct_predictions = np.zeros(10)

        for j in range(len(X)):
            Y = np.zeros(num_classes)
            Y[y[j]] = 1

            yhat = Model.forward(X[j], is_train=True)
            Model.backward(yhat, Y)

            #DEBUG
            if np.argmax(yhat) == y[j]:
                correct_predictions[y[j]] += 1
            #TODO: how to avoid log of 0
            J -= np.sum(Y * np.log(yhat) + (1 - Y) * np.log(1 - yhat))

            if (j + 1) % batch_size == 0:
                Model.update_weights(learning_rate)

            # if (j+1) % 1000 == 0:
            #     print "Cost in ", j+1, " iterations = ", J/(j+1)
            #     print "Correct predictions count ", correct_predictions
            #     print "Total correct ", np.sum(correct_predictions), "/", (j+1)

        Model.update_weights(learning_rate)
        
        #Checking against cross validation set to prevent overfitting
        cross_val_prediction_probabilities = predict(Model, X_val)
        cross_val_predictions = np.argmax(cross_val_prediction_probabilities, axis=1)
        prediction_accuracy = sum(cross_val_predictions == y_val) / len(y_val)

        print "Epoch: ", i+1
        print "Training prediction accuracy: %.3f" % (np.sum(correct_predictions)/len(X))
        print "Cross validation accuracy: ", prediction_accuracy

    return Model

def predict(Model, X_test):
    predictions = []
    for i in range (len(X_test)):
        probs = Model.forward(X_test[i])
        predictions.append(probs)

    return predictions

