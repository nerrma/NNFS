# Optimiser Implementations
# - SGD
# - Adagrad
# - RMSprop
# - Adam

# SGD implemetation
class SGD:

    # Constructor with settings
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum
    
    # Call before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            # Initialise momentums if not found
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
                weight_updates = self.momentum * layer.weight_momentums - self.current_lr * layer.dweights
                layer.weight_momentums = weight_updates

                bias_updates = self.momentum * layer.bias_momentums - self.current_lr * layer.dbiases
                layer.bias_momentums = bias_updates 
            else:
                weight_updates = -self.current_lr * layer.dweights
                bias_updates = -self.current_lr * layer.dbiases

            layer.weights += weight_updates
            layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

