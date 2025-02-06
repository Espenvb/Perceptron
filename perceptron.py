import numpy as np
import Functions.loss_functions as loss_functions
import Functions.activation_functions as activation_functions
import Functions.regularization_functions as regularization_functions
import random

class Perceptron:
    def __init__(self, 
                 regularization=None, 
                 l=0.001, 
                 epochs=5,
                 activation='ReLu', 
                 loss_function='MSE', 
                 learning_rate=0.1, 
                 batch_size=32):
        '''
        Inputs:
            - regularization=None:
                The regularization function to use. If None, no regularization is used.
                Valid values are 'l1' and 'l2'.
            - l=0.001:
                The regularization parameter.
            - epochs=5:
                The number of epochs to train the perceptron.
            - activation='ReLu':
                The activation function to use.
                Valid values are 'ReLu'.
            - loss_function='MSE':
                The loss function to use.
                Valid values are 'MSE'.
            - learning_rate=0.1:
                The learning rate to use.
            - batch_size=32:
                The batch size to use.
        '''
        
        valid_regularization_functions = {name for name in dir(regularization_functions)
                        if callable(getattr(regularization_functions, name, None))}
        if regularization not in valid_regularization_functions and regularization is not None:
            raise ValueError(f"Invalid value: {regularization}. Allowed values are {valid_regularization_functions}")
        self.regulartization = getattr(regularization_functions, regularization)
        
        valid_loss_functions = {name for name in dir(loss_functions)
                        if isinstance(getattr(loss_functions, name, None), type)}
        if loss_function not in valid_loss_functions:
            raise ValueError(f"Invalid Value: {loss_function}. Allowed values are {valid_loss_functions}")
        self.loss_function = getattr(loss_functions, loss_function)
        
        valid_activation_functions = {name for name in dir(activation_functions)
                        if isinstance(getattr(activation_functions, name, None), type)}
        if activation not in valid_activation_functions:
            raise ValueError(f"Invalid Value: {activation}. Allowed values are {valid_activation_functions}")
        self.activation = getattr(activation_functions, activation)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.l = l 
        self.prev_output = 0
        self.prev_sum = 0
        self.weights = np.array([0])
        self.bias = 0
        
    def forward(self, x):
        '''
        #Forward pass of the preceptron
        
        Inputs:
        
        x - input data
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
        if self.regulartization == None:
            error = self.loss_function.derivative(y, self.prev_output)   
        else:
            error = self.loss_function.derivative(y, self.prev_output) + self.regulartization(self.weights, self.l)
        update = self.learning_rate*error 
        self.weights -= update*np.sum(x, axis=0)
        self.bias -= np.sum(update, axis=0)
        return self.loss_function.function(y, self.prev_output)
    
    def train(self, x, y): 
        '''
        Inputs:
            x - input data, 1D or 2D NumPy array
            
            y - gold value 1D NumPy array
            
            Trains the preceptron using the input data and gold values for n epochs,specified in the constructor.
        ''' 
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if x.ndim > 2 or x.ndim == 0:
            raise TypeError("X must be 1D or 2D data.")
        if y.ndim > 1 or x.ndim == 0:
            raise TypeError("Y must be 1D data.")
        
        if x.ndim == 1:
            self.input_size = x.shape[0]
            self.training_size = 1
        elif x.ndim == 2:
            self.input_size = x.shape[1]
            self.training_size = x.shape[0]
            if self.training_size != y.shape[0]:
                raise ValueError("X and Y must have compatible sizes.")
            
        print("Initialization Done.")
        print(f"    -Input Size: {self.input_size}")
        print(f"    -Training Samples: {self.training_size}")
        
        self.weights = np.random.uniform(0, 1, self.input_size)
        self.bias = random.uniform(0, 1) 
        
        for e in range (0, self.epochs):
            training_set = np.column_stack((x, y))
            
            np.random.shuffle(training_set)
            
            x_shuffled = training_set[:,:-1]
            y_shuffled = training_set[:,-1]

            for i in range (0, self.training_size, self.batch_size):        
                end_index = min(i + self.batch_size, self.training_size)
                x_batch = x_shuffled[i:end_index]
                y_batch = y_shuffled[i:end_index]
                
                self.forward(x_batch)
                self.backward(x_batch, y_batch)
            print("Epoch ", e, " Done.")
            print("Epoch Loss: ", self.loss_function.function(y_shuffled, self.forward(x_shuffled)))
            print("-------------------------------------------------")   
            
    def test(self, x, y):
        z = self.forward(x)
        loss = self.loss_function.function(z, y)
        print("Total Loss: ", loss)
                
        



    
        
        
    
