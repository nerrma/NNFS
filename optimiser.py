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


# Adagrad optimiser
class Adagrad:
    
    # Constructor
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
    # Call before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If there is already no cache, initialise one
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2
        
        # Update weights and biases using SGD with cache
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# RMSprop optimiser
class RMSprop:

    # Constructor
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho

    # Call before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If there is already no cache, initialise one
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
        
        # Update cache
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2
        
        # Update weights and biases using SGD with cache
        layer.weights += -self.current_lr * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_lr * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Adam optimiser
class Adam:

    # Constructor
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # Call before parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_lr = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        # If there is already no cache, initialise one
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)
      
        # Update momentums with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        
        # Update cache
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2 
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2 

        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Update weights and biases using SGD with cache
        layer.weights += -self.current_lr * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_lr * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1
