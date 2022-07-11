import numpy as np

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, loss, loss_prime, x_train, y_train, x_val=None, y_val=None, epochs=1000, learning_rate=0.01, batch_size=32, verbose=True):
        for e in range(epochs):
            error = 0
            # batches = self.create_batches(x_train, y_train, batch_size)
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # loss
                error += loss(y, output)

                # backward
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)

            # Predict with network on x_train
            train_acc = 0
            for x, y in zip(x_train, y_train):
                train_pred = self.predict(x)
                train_acc += (1/len(x_train)) * 1 if np.argmax(train_pred) == np.argmax(y) else 0
            
            # Predict with network on x_val
            val_acc = 0
            if x_val is not None and y_val is not None:
                for x, y in zip(x_train, y_train):
                    val_pred = self.predict(x)
                    val_acc += (1/len(x_val)) * 1 if np.argmax(val_pred) == np.argmax(y) else 0

            if verbose and x_val is not None and y_val is not None:
                print(f'Epoch: {e + 1}/{epochs}, Loss: {error}, Train Acc: {train_acc}, Val Acc: {val_acc}')
            elif verbose:
                print(f'Epoch: {e + 1}/{epochs}, Loss: {error}, Train Acc: {train_acc}')

    def create_batches(self, x, y, batch_size):
            m = x.shape[0]
            num_batches = m / batch_size
            batches = []
            for i in range(int(num_batches+1)):
                batch_x = x[i*batch_size:(i+1)*batch_size]
                batch_y = y[i*batch_size:(i+1)*batch_size]
                batches.append((batch_x, batch_y))
            
            # without this, batch sizes that are perfectly divisible will create an 
            # empty array at index -1
            if m % batch_size == 0:
                batches.pop(-1)

            return batches

    def compute_loss(Y, Y_hat):
        m = Y.shape[0]
        L = -1./m * np.sum(Y * np.log(Y_hat))
        return L