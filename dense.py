import numpy as np
from optimizers import Optimizers
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # Initialize a random weight matrix.
        self.bias = np.random.randn(output_size, 1)             # Initialize a random bias vector.
        
    def forward(self, input):
        self.input = input                                      # Our input matrix (X).
        return np.dot(self.weights, self.input) + self.bias     # Performs matrix multiplication/addition to forward propogate.

    def backward(self, output_error, learning_rate):
        # Note X = input, Y = output. W = weights, B = biases.
        # Here, we're performing gradient descent and backpropogation to update the weights and biases.
        weights_error = np.dot(output_error, self.input.T)      # dE/dW (dE/dY * X.T)
        input_error = np.dot(self.weights.T, output_error)      # dE/dX (W.T * dE/dY)

        # opt = Optimizers(self.weights.size, learning_rate)
        # self.weights = np.reshape(opt.Adam(self.weights, weights_error), self.weights.shape)
        # self.bias = np.reshape(opt.Adam(self.bias, output_error), self.bias.shape)
        self.weights -= learning_rate * weights_error           # Performing gradient descent with our learning rate.
        self.bias -= learning_rate * output_error              # Note that dE/dB = dE/dY !!
        return input_error

    