import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation                        # The activation function used.
        self.activation_prime = activation_prime            # Derivative of the activation function (for backprop).

    def forward(self, input):
        self.input = input                                  # Our input matrix (X).
        return self.activation(self.input)                  # g(X) where g is our activation function.

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))  # dE/dX = dE/dY * g`(X). 