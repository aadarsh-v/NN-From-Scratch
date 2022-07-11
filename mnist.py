import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from activations import Tanh, Sigmoid, ReLu, Softmax
from loss import categorical_cross_entropy, categorical_cross_entropy_prime, mse, mse_prime
from model import Model

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

# neural network
model = Model()
model.add(Dense(28 * 28, 40))
model.add(Sigmoid())
model.add(Dense(40, 10))
model.add(Softmax())

# train
model.train(categorical_cross_entropy, categorical_cross_entropy_prime, x_train, y_train, epochs=100, learning_rate=0.1)

# test
accuracy = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
    accuracy += (100/len(y_test)) * 1 if np.argmax(output) == np.argmax(y) else 0

print("Test Accuracy: " + str(accuracy))