import nnfs
import numpy as np
from nnfs.datasets import spiral_data

import accuracy
import activation
import layer
import loss
import model
import optimiser

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = model.Model()

model.add(layer.Dense(2, 512, [-1, 5e-4], [-1, 5e-4]))
model.add(activation.ReLU())
model.add(layer.Dropout(0.1))
model.add(layer.Dense(512, 3, [-1, -1], [-1, -1]))
model.add(activation.Softmax())

model.set(loss=loss.CategorialCrossEntropy(),
          optimiser=optimiser.Adam(learning_rate=0.05, decay=5e-5),
          accuracy=accuracy.Categorial()
         )

model.finalise()

model.train(X, y, validation_data=(X_test, y_test), epochs=100, print_every=1)

