import numpy as np
from loss_function import MSE
from activation_function import ReLu

class Perceptron:
    def __init__(self, 
                 regularization=None, 
                 l=0.001, 
                 epochs=5,
                 activation=ReLu, 
                 loss_function=MSE, 
                 learning_rate=0.1, 
                 batch_size=32):
        regularization_values = [None, "l1", "l2"]
        if regularization not in regularization_values:
            raise ValueError(f"Invalid value: {regularization}. Allowed values are {regularization_values}")
        self.regulartization = regularization
        
        loss_functions = {name: getattr(loss_function, name) for name in dir(loss_function) 
            if callable(getattr(loss_function, name))}
        if loss_function not in loss_functions:
            raise ValueError(f"invalid value: {loss_function}. Allowed Values are {loss_functions}")
        
        
        self.activation = activation
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.prev_output = 0
        self.prev_sum = 0
        self.weights = 0
        self.bias = 0
        
    def forward(self, x):
        '''
        #Forward pass of the preceptron
        
        Inputs:
        
        x - input data
        
        Does only suport one traning example at the time 
        
        Need to implement batching later
        '''
        s = np.matmul(self.weights, np.transpose(x)) + self.bias
        self.prev_sum = s
        z = self.activation.function(s)  
        self.prev_output = z
        return z
    
    def backward(self, x, y):
        '''
            Inptus:
            
            x - input data, one training sample
            
            y - gold value, The true output value
            
            Updates the weights of the preceptron using the backproagation algorithm with gradient decent,
            and returns the loss.
        '''
        error = self.loss_function.derivative(y, self.prev_output)   
        update = self.learning_rate*error  
        self.weights -= update*np.sum(x, axis=0)
        self.bias -= np.sum(update, axis=0)
        return self.loss_function.function(y, self.prev_output)
    
    def predict(self, x):
        s = np.matmul(self.weights, x) + self.bias
        return self.activation(s)
    
    def train(self, x, y):
        data_length = len(x)
        for i in range (0, data_length, self.batch_size):
            end_index = min(i + self.batch_size, data_length)
            x_batch = x[i:end_index]
            y_batch = y[i:end_index]
            self.forward(x_batch)
            print("Loss: ", self.backward(x_batch, y_batch))
            print("W", self.weights)
            print("b", self.bias)
    
        
        
    
