# imlementing a layer class
class DenseLayer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons  # initialize as batch size
        # self.activation = activation   # sigmoid

        self.z = None

    def forwardprop(self, input_data, weights=None, biases=None):
        # weight matrix of every datapoint is matrix of (n_features, num_neur)

        # input data is one datapoint here
        if weights is None and biases is None:
            self.weights = np.random.randn(input_data.shape[0], self.num_neurons)
            self.biases = np.random.randn(self.num_neurons)
            weights = self.weights
            biases = self.biases
            self.z = np.dot(weights.T, input_data) + biases  # for every datapoint
        else:
            self.z = np.dot(weights.T, input_data) + biases  # for every datapoint

        # if self.activation is not None:  # just for sigmoid
        # return self.sigmoid(self.z)
        # return self.weights, self.biases,self.z
        return self.z

    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases


class DenseNetwork:
    def __init__(self, batch_size, num_epoch, layers, alpha):
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.layers = layers
        self.z_1 = None
        self.z_2 = None
        self.z_3 = None
        self.a_1 = None
        self.a_2 = None
        self.a_3 = None
        self.w_3 = None
        self.w_2 = None
        self.w_1 = None
        self.b_3 = None
        self.b_2 = None
        self.b_1 = None
        self.alpha = alpha  # as learning rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input_data):
        # z initialization for every layer
        layer_1 = self.layers[0]
        layer_2 = self.layers[1]
        layer_3 = self.layers[2]
        # for every datapoint
        self.z_1 = layer_1.forwardprop(input_data)
        # print(self.z_1)
        self.a_1 = self.sigmoid(self.z_1)
        self.z_2 = layer_2.forwardprop(self.a_1)
        self.a_2 = self.sigmoid(self.z_2)
        self.z_3 = layer_3.forwardprop(self.a_2)
        self.a_3 = self.sigmoid(self.z_3)

    def backpropagation(self, input_data, z_1, z_2, z_3, a_1, a_2, a_3, y):

        # y is also one datapoint

        # suppose Cross Entropy loss

        # (dL/a3 * da3/dz3) *dz3/dw3
        grad_W_3 = -y * np.dot((1 - a_3), a_2.T) - ((1 - y) * np.dot(a_3, a_2.T))

        # (dL/da3 * da3/dz3) * dz3/da2 * da2/dz2 * dz2/dw2
        gard_W_2 = np.dot(self.w_3.T, a_2, (1 - a_2), (-y * a_3 - ((1 - y) * (1 - a_3))), a_2.T)

        # (dL/da3 * da3/dz3 * dz3/da2 * da2/dz2) * dz2/da1 * da1/dz1*dz1/dw1
        gard_W_1 = np.dot(self.w_3.T, a_2, (1 - a_2), (-y * a_3 - ((1 - y) * (1 - a_3))), self.w_2.T, a_1, (1 - a_1),
                          a_1.T)

    def call(self, X, y=None, training=False):
        n = X.shape[0]
        num_batch = n // self.batch_size
        for j in range(self.num_epoch):
            for i in range(num_batch):
                start_index = i * self.batch_size
                end_index = start_index + self.batch_size
                batch_data = X[start_index:end_index]  # iloc for dataframes
                batch_data_y = y[start_index:end_index]

            if training == False:

                for i in range(batch_data.shape[0]):
                    self.feedforward(batch_data[i])
                    self.w_1 = self.layers[0].get_weights()
                    self.w_2 = self.layers[0].get_weights()
                    self.w_3 = self.layers[0].get_weights()
                    self.b_1 = self.layers[0].get_biases()
                    self.b_2 = self.layers[0].get_biases()
                    self.b_3 = self.layers[0].get_biases()
            else:
                # back propagation
                for i in range(batch_data.shape[0]):
                    # for every datapoint
                    z_1 = self.z_1
                    z_2 = self.z_2
                    z_3 = self.z_3
                    a_1 = self.a_1.reshape(self.a_1.shape[0], 1)
                    a_2 = self.a_2.reshape(self.a_2.shape[0], 1)
                    a_3 = self.a_3.reshape(self.a_3.shape[0], 1)

                    grad_W_3 = self.backpropagation(batch_data, z_1, z_2, z_3, a_1, a_2, a_3, batch_data_y[i])
                    grad_W_2 = self.backpropagation(batch_data, z_1, z_2, z_3, a_1, a_2, a_3, batch_data_y[i])
                    grad_W_1 = self.backpropagation(batch_data, z_1, z_2, z_3, a_1, a_2, a_3, batch_data_y[i])
                    self.w_3 -= self.alpha * grad_W3
                    self.w_2 -= self.alpha * grad_W2
                    self.w_1 -= self.alpha * grad_W1
                    self.b_3 -= -(batch_data_y[i] * 1 / a_3 + ((1 - batch_data_y[i]) * (1 - a_3)))  ###
                    self.b_2 -= -(batch_data_y[i] * 1 / a_2 + ((1 - batch_data_y[i]) * (1 - a_2)))
                    self.b_1 -= -(batch_data_y[i] * 1 / a_1 + ((1 - batch_data_y[i]) * (1 - a_1)))
        return self.sigmoid(self.layers[2].forwardprop(self.a_2, weights=self.w_3, biases=self.b_3))  # as y_predicted




import numpy as np
# suppose we have 3 layers
batch_size = 32
layers = [DenseLayer(batch_size), DenseLayer(batch_size), DenseLayer(1)]
