import layer

# Model class to store models easily
class Model:
    def __init__(self):
        self.layers = []
    
    # Add a layer
    def add(self, layer):
        self.layers.append(layer)

    # Set a form of loss and optimisation
    def set(self, *, loss, opti):
        self.loss = loss
        self.optimiser = opti

    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in range(1, epochs+1):
            output = self.forward(X)
    
    def finalise(self):
        self.input_layer = layer.input()
        layer_count = len(self.layers)

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

    def forward(self, X):
        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output
