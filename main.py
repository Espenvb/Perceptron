import numpy as np
from perceptron import Perceptron

model = Perceptron(l=0.001, epochs=5, activation='RLu', loss_function='MSE', learning_rate=0.1, batch_size=32)

model.train(np.array([1, 2, 4]), np.array([2, 1]))