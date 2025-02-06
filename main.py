import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt


model = Perceptron(regularization='l1', l=0.001, epochs=10, activation='Heaviside', loss_function='SE', learning_rate=0.1, batch_size=1)

class_1 = np.random.multivariate_normal([1, 1], [[1,0],[0,1]], 50)
class_1_lables = np.ones(50)
class_2 = np.random.multivariate_normal([-1, -1], [[1,0],[0,1]], 50)
class_2_lables = np.zeros(50)
plt.scatter(class_1[:,0], class_1[:,1], c='r')
plt.scatter(class_2[:,0], class_2[:,1], c='b')
plt.show()

x_train = np.concatenate((class_1[:40], class_2[:40]))
y_train = np.concatenate((class_1_lables[:40], class_2_lables[:40]))

x_test = np.concatenate((class_1[40:49], class_2[40:49]))
y_test = np.concatenate((class_1_lables[40:49], class_2_lables[40:49]))

model.train(x_train, y_train)

model.test(x_test, y_test)
weights = model.weights
bias = model.bias

x = np.linspace(-5, 5, 100)
y = (-weights[0]*x - bias)/weights[1]
plt.plot(x, y)
plt.scatter(class_1[:,0], class_1[:,1], c='r')
plt.scatter(class_2[:,0], class_2[:,1], c='b')
plt.show()
