import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import ReLu, Sigmoid, Tanh, Softmax
from loss import binary_cross_entropy, binary_cross_entropy_prime, mse, mse_prime, categorical_cross_entropy, categorical_cross_entropy_prime
from model import Model

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    # x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = np_utils.to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

# neural network
model = Model()
model.add(Convolutional((1, 28, 28), 3, 3))
model.add(Sigmoid())
model.add(Reshape((3, 26, 26), (3 * 26 * 26, 1)))
model.add(Dense(3 * 26 * 26, 40))
model.add(Sigmoid())
model.add(Dense(40, 10))
model.add(Softmax())

# train
model.train(
    categorical_cross_entropy,
    categorical_cross_entropy_prime,
    x_train,
    y_train,
    epochs=100,
    learning_rate=0.1
)

# test
accuracy = 0
for x, y in zip(x_test, y_test):
    output = model.predict(x)
    # print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))
    accuracy += (100/len(y_test)) * 1 if np.argmax(output) == np.argmax(y) else 0

print("Test Accuracy: " + str(accuracy))