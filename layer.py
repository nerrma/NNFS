import numpy as np

# Class to represent a dense layer
class Dense:
    # Constructor (w_reg_lambda, b_reg_lambda = [L1, L2])
    def __init__(self, n_inputs, n_neurons, w_reg_lambda, b_reg_lambda):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.w_reg_lambda = w_reg_lambda
        self.b_reg_lambda = b_reg_lambda

    # Forward pass through layer
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass (d stands for derivative)
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # L1 Reg Backprop
        if self.w_reg_lambda[0] > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.w_reg_lambda[0] * dL1
        
        if self.b_reg_lambda[0] > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.b_reg_lambda[0] * dL1

        # L2 Reg Backprop
        if self.w_reg_lambda[1] > 0:
            self.dweights += 2 * self.w_reg_lambda[1] * self.weights
        
        if self.b_reg_lambda[1] > 0:
            self.dbiases += 2 * self.b_reg_lambda[1] * self.biases
        
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases


# Dropout layer
class Dropout:
    def __init__(self, rate):
        self.rate = 1 - rate

    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return

        # Generate random zeros for dropout step
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        # Perform dropout
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask

# Input layer (for training implementation, only stores training data)
class Input:
    def forward(self, inputs, training):
        self.output = inputs
