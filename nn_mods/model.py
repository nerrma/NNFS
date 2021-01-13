import pickle

import numpy as np

from nn_mods import activation, layer, loss


# Model class to store models easily
class Model:
    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
    
    # Add a layer
    def add(self, layer):
        self.layers.append(layer)

    # Set a form of loss and optimisation
    def set(self, *, loss, optimiser, accuracy):
        if loss is not None:
            self.loss = loss

        if optimiser is not None:
            self.optimiser = optimiser

        if accuracy is not None:
            self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            output = self.forward(X, training=True)
            data_loss, reg_loss = self.loss.calculate(output, y, include_reg=True)
            loss = data_loss + reg_loss
            
            predictions = self.output_layer_activation.predictions(output)
            save_predictions = predictions
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            self.optimiser.pre_update_params()
            for layer in self.trainable_layers:
                self.optimiser.update_params(layer)

            self.optimiser.post_update_params()

            # Print a summary
            if not epoch % print_every or epoch == 1:
                print(f'epoch: {epoch}, ' +
                    f'acc: {accuracy:.3f}, ' +
                    f'loss: {loss:.3f} (' +
                    f'data_loss: {data_loss:.3f}, ' +
                    f'reg_loss: {reg_loss:.3f}), ' +
                    f'lr: {self.optimiser.current_lr}')

        if validation_data is not None:
            X_val, y_val = validation_data

            output = self.forward(X_val, training=False)
            loss = self.loss.calculate(output, y_val)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y_val)

            # Print a summary
            print(f'validation, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')
    
        return save_predictions 
    
    def finalise(self):
        self.input_layer = layer.Input()
        layer_count = len(self.layers)
    
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], activation.Softmax) and isinstance(self.loss, loss.CategorialCrossEntropy):
            self.softmax_classifier_output = Softmax_Loss_CatCross()

    def forward(self, X, training):
        self.input_layer.forward(X, training)

        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        return layer.output

    def backward(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        self.loss.backward(output, y)

        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_params(self):
        parameters = []
        
        for layer in self.trainable_layers:
            parameters.append(layer.get_params())

        return parameters

    def set_params(self, params):
        for parameter_set, layer in zip(params, self.trainable_layers):
            layer.set_params(*parameter_set)
        
    def save_params(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.get_params(), f)
    
    def load_params(self, path):
        with open(path, 'rb') as f:
            self.set_params(pickle.load(f))

class Softmax_Loss_CatCross():
    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples
        


