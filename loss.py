# Implementation of Loss
import numpy as np

# Mean Loss (Mean of sample losses)
class Mean:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorial Cross Entropy Loss 
class CategorialCrossEntropy(Mean):
    
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        if len(y_true.shape) == 1:
            correct_conf = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_conf = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log = -np.log(correct_confidences)
        return neg_log


