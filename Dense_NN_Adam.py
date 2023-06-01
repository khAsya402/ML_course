import numpy as np
class DenseLayer:
    def __init__(self, input_size, output_size, activation=None, T=4):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros(output_size)
        self.activation = activation
        self.T = T
        self.output_size = output_size

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

        return self.output

    def backward(self, grad_output, learning_rate):
        ro_1 = 0.9
        ro_2 = 0.999
        delta = 10 ** -8
        t = 0
        while t <= self.T:
            # s = 0
            # r = 0
            s = [np.zeros_like(param) for param in self.weights]
            r = [np.zeros_like(param) for param in self.weights]
            s_b = [np.zeros_like(param) for param in self.biases]
            r_b = [np.zeros_like(param) for param in self.biases]
            grad_weights = [np.zeros_like(param) for param in self.weights]
            grad_biases = [np.zeros_like(param) for param in self.biases]
            for i in range(self.output_size):
                s[i] = (ro_1 * s[i] + (1 - ro_1) * grad_output[i]) / (1 - ro_1 ** t + 1)
                r[i] = (ro_2 * r[i] + (1 - ro_2) * grad_output[i] * grad_output[i]) / (1 - ro_2 ** t + 1)
                s_b[i] = (ro_1 * s_b[i] + (1 - ro_1) * np.sum(grad_output)) / (1 - ro_1 ** t + 1)
                r_b[i] = (ro_2 * r_b[i] + (1 - ro_2) * np.sum(grad_output) ** 2) / (1 - ro_2 ** t + 1)
                # grad_weights = np.dot(self.inputs.T, grad_output)
                grad_weights[i] = s[i] / (np.sqrt(r[i]) + delta)
                grad_biases[i] = s_b[i] / (np.sqrt(r_b[i]) + delta)

                if self.activation is None:
                    grad_input = np.dot(grad_output, self.weights.T)
                elif self.activation == 'sigmoid':
                    grad_input = self.sigmoid(np.dot(grad_output, self.weights.T))
                elif self.activation == 'relu':
                    grad_input = self.relu(np.dot(grad_output, self.weights.T))

                self.weights[i] -= learning_rate * grad_weights[i]
                self.biases[i] -= learning_rate * grad_biases[i]
            t += 1
        return grad_input

class DenseNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    # def backward(self, grad_output, learning_rate,T = 1000):
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            # for t in range(T):
            grad_output = layer.backward(grad_output, learning_rate)


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

# Generate synthetic dataset
X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Standardize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train scikit-learn's LinearRegression model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Predict with scikit-learn's LinearRegression model
y_pred_lr = lr_model.predict(X_test_scaled)

# Train the DenseNetwork implemented from scratch
dense_net = DenseNetwork()
dense_net.add_layer(DenseLayer(10, 10))
dense_net.add_layer(DenseLayer(10, 1))

# Train the DenseNetwork using gradient descent
learning_rate = 0.001

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    y_pred = dense_net.forward(X_train_scaled)

    # Compute loss (mean squared error)
    loss = np.mean((y_pred - y_train) ** 2)
    #     print(f'epoch {epoch}:{loss}')
    # Backward pass
    grad_output = 2 * (y_pred - y_train) / len(X_train_scaled)
    dense_net.backward(grad_output, learning_rate)

# Predict with the DenseNetwork
y_pred_dense = dense_net.forward(X_test_scaled)

