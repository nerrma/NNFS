# Contains useful activation functions
# - ReLU (Rectified Linear)
# - Softmax
# - Sigmoid

import numpy as np

# ReLU activation
class ReLU:
    # Forward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        # ReLU through using maximum
        self.output = np.maximum(0,inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs):
        return outputs

# Softmax activation
class Softmax:
    # Foward pass
    def forward(self, inputs, training):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites
    
    # Backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)

        for i, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[i] = np.dot(jacobian, single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

# Sigmoid activation
class Sigmoid:
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = 1/(1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
    
    def predictions(self, outputs):
        return (outputs > 0.5) * 1


