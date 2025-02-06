import numpy as np
import loss_function
from activation_function import ReLu
from perceptron import Perceptron

print()

train = np.array([
    [1, 1, 0.08, 0.72, 1],
    [2,1,0.1,1,1],
    [3,1,0.26,0.58,1],
    [4,1,0.35,0.95,0],
    [5,1,0.45,0.15,1],
    [6,1,0.6,0.3,1],
    [7,1,0.7,0.65,0],
    [8,1,0.92,0.45,0]
])

test = np.array([
    [9,1,0.42,0.85,0],
    [10,1,0.65,0.55,0],
    [11,1,0.2,0.3,1],
    [12,1,0.2,1,0],
    [13,1,0.85,0.1,1]
])

x_train = train[:,2:-1]
y_train = train[:, -1]

x_test = test[:,2:-1]
y_test = test[:, -1]


init_weights = np.array([0])
activation_function = ReLu()
bias = 0.1
weights = np.array([0.1, 0.1])

model = Perceptron(loss_function='MSE')

x = np.array([2,1])
y = np.array([2])

model.train(x_train, y_train)

x = np.array([3,2])
y = np.array([1])

#model.train(x, y)


