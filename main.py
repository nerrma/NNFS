import matplotlib.pyplot as plt
import nnfs
import numpy as np
from nnfs.datasets import spiral_data
import os

from nn_mods import accuracy, activation, layer, loss, model, optimiser

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

parameters = model.get_params()

if os.path.exists('model_params'):
    model.load_params('model_params')

output = model.train(X, y, validation_data=(X_test, y_test), epochs=50, print_every=10)
model.save_params('model_params')

# output = output.reshape(3, 3000)
print(output)


#plt.contourf(X[:, 0], X[:, 1], Z, levels=output, cmap='brg')
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c=y, s=40, cmap='brg')
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, c=output, s=40, cmap='brg')
plt.show()


