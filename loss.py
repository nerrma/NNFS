# Implementation of Loss
import numpy as np


# General Loss 
class Loss:
    
    # Mean Loss
    def calculate(self, output, y, *, include_reg=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        if not include_reg:
            return data_loss

        return data_loss, self.reg_loss()
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    # Regularisation Loss
    def reg_loss(self):
        reg_loss = 0
        
        for layer in self.trainable_layers:
            # L1 Regularisation
            if layer.w_reg_lambda[0] > 0:
                reg_loss += layer.w_reg_lambda[0] * np.sum(np.abs(layer.weights))

            if layer.b_reg_lambda[0] > 0:
                reg_loss += layer.b_reg_lambda[0] * np.sum(np.abs(layer.biases))

            # L2 Regularisation
            if layer.w_reg_lambda[1] > 0:
                reg_loss += layer.w_reg_lambda[1] * np.sum(layer.weights * layer.weights)

            if layer.b_reg_lambda[1] > 0:
                reg_loss += layer.b_reg_lambda[1] * np.sum(layer.biases * layer.biases)

        return reg_loss

# Categorial Cross Entropy Loss 
class CategorialCrossEntropy(Loss):
    
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_conf = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_conf = np.sum(y_pred_clipped*y_true, axis=1)
        
        neg_log = -np.log(correct_conf)
        return neg_log

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])

        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.inputs /= samples

# Binary Cross Entropy Loss
class BinaryCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        ouputs = len(dvalues[0])

        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
        
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        self.dinputs /= samples

# Mean Squared Error Loss
class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs /= samples

# Mean Absolute Error Loss
class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        self.dinputs =  np.sign(y_true - dvalues) / outputs
        self.dinputs /= samples
