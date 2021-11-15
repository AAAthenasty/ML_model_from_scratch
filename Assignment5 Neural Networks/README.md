# Introduction
This program implements a neural network to classify images using a single hidden layer neural network.\
In addition, I implement Adagrad, a variant of stochastic gradient descent, in this model.

This neural network implement a single-hidden-layer neural network with a sigmoid activation function for the hidden layer, and a softmax on the output layer. 

# Initialization
In order to use a deep network, the weights and biases in the network must be initalized. \
This is typically done with a random initialization, or initializing the weights from some other training procedure. \

In this model, these two initialization methods are both accepted through the paramenter parsed in:
* RANDOM:The weights are initialized randomly from a uniform distribution from -0.1 to 0.1. The bias parameters are initialized to zero.\
* ZERO:All weights are initialized to 0.\

# NeuralNet model
neuralnet.py implements an optical character recognizer using a one hidden layer neural network with sigmoid activations. \
The program learn the parameters of the model on the training data, report the cross-entropy at the end of each epoch on both train and validation data, \
and at the end of training write out its predictions and error rates on both datasets.

This model satisfies the following requirements:

* Use a sigmoid activation function on the hidden layer and softmax on the output layer to ensure it forms a proper probability distribution.
* Number of hidden units for the hidden layer should be determined by a command line flag.
* Support two different initialization strategies, as described in Section 7.1, selecting between them via a command line flag.
* Use stochastic gradient descent (SGD) to optimize the parameters for one hidden layer neural network.
* The number of epochs will be specified as a command line flag.
* Set the learning rate via a command line flag.
