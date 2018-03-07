# Implement a convolutional neural network to recognize digits

A simple neural network with 1 hidden layer is implemented in nnet.py.

A convolutional neural network is implemented in cnn.py with convolution and ReLU layer

CNN hyperparameters can be adjusted in cnn.py

Output for NIST dataset with various hyperparams
-------------------------------------------------------------------------------
1.
```
Model = model.model()
Model.add(layers.convolution_layer(field=8, padding=0, stride=1, depth=1, filters=5))
Model.add(layers.relu_layer())
Model.add(layers.flatten_layer())
Model.add(layers.linear_layer(13*13*5, 10))
Model.set_loss(layers.softmax_cross_entropy())
batch_size = 20
learning_rate = 0.01
epochs = 20
num_classes = 10
dropout_rate = 0.00
```
### Output
```
Epoch:  20
Training prediction accuracy: 1.000
Cross validation accuracy:  0.926
Test set accuracy =  0.948
```
-------------------------------------------------------------------------------
2. FEWER NUMBER OF EPOCHS
```
Model = model.model()
Model.add(layers.convolution_layer(field=8, padding=0, stride=1, depth=1, filters=5))
Model.add(layers.relu_layer())
Model.add(layers.flatten_layer())
Model.add(layers.linear_layer(13*13*5, 10))
Model.set_loss(layers.softmax_cross_entropy())
batch_size = 20
learning_rate = 0.01
epochs = 10
num_classes = 10
dropout_rate = 0.00
```
### Output
```
Epoch: 10
Training prediction accuracy: 0.985
Cross validation accuracy:  0.924
Test set accuracy =  0.938
```
-------------------------------------------------------------------------------
3. WITH DROPOUT
```
Model = model.model()
Model.add(layers.convolution_layer(field=8, padding=0, stride=1, depth=1, filters=5))
Model.add(layers.relu_layer())
Model.add(layers.flatten_layer())
Model.add(layers.dropout_layer(r = dropout_rate))
Model.add(layers.linear_layer(13*13*5, 10))
Model.set_loss(layers.softmax_cross_entropy())
batch_size = 20
learning_rate = 0.01
epochs = 20
num_classes = 10
dropout_rate = 0.1
```
### Output
```
Epoch:  20
Training prediction accuracy: 0.997
Cross validation accuracy:  0.941
Test set accuracy =  0.951
```
-------------------------------------------------------------------------------
4. MULTINOMIAL LOGISTIC REGRESSION
```
Model = model.model()
Model.add(layers.flatten_layer())
Model.add(layers.dropout_layer(r = dropout_rate))
Model.add(layers.linear_layer(20*20, 10))
Model.set_loss(layers.softmax_cross_entropy())
batch_size = 20
learning_rate = 0.01
epochs = 20
num_classes = 10
```
### Output
```
Epoch:  20
Training prediction accuracy: 0.950
Cross validation accuracy:  0.903
Test set accuracy =  0.9
```
-------------------------------------------------------------------------------
5. 3 LAYER NEURAL NETWORK
```
Model = model.model()
Model.add(layers.flatten_layer())
Model.add(layers.linear_layer(20*20, 200))
Model.add(layers.dropout_layer(r = dropout_rate))
Model.add(layers.linear_layer(200, 10))
Model.set_loss(layers.softmax_cross_entropy())
batch_size = 20
learning_rate = 0.01
epochs = 30
num_classes = 10
dropout_rate = 0.1
```
### Output
```
Epoch:  30
Training prediction accuracy: 0.963
Cross validation accuracy:  0.888
Test set accuracy =  0.873
```
(Neural network easily overfits while the test accuracy is lesser than training by 9%)