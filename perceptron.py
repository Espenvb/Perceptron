import numpy as np
import loss_functions
import activation_functions
import regularization_functions

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
        self.regulartization = regularization
        
        valid_loss_functions = {name for name in dir(loss_functions)
                        if isinstance(getattr(loss_functions, name, None), type)}
        if 'MSE' not in valid_loss_functions:
            raise ValueError(f"Invalid Value: {loss_function}. Allowed values are {valid_loss_functions}")
        self.loss_function = getattr(loss_functions, "MSE")
        
        valid_activation_functions = {name for name in dir(activation_functions)
                        if isinstance(getattr(activation_functions, name, None), type)}
        if 'MSE' not in valid_loss_functions:
            raise ValueError(f"Invalid Value: {activation}. Allowed values are {valid_activation_functions}")
        self.activation = getattr(loss_functions, "MSE")
        
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
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if x.ndim > 2 or x.ndim == 0:
            raise TypeError("Input must be 1D or 2D data.")
        
        if x.ndim == 1:
            self.input_size = x.shape[0]
        elif x.ndim == 2:
            self.input_size = x.shape[1]
        
        for i in range (0, data_length, self.batch_size):
            end_index = min(i + self.batch_size, data_length)
            x_batch = x[i:end_index]
            y_batch = y[i:end_index]
            self.forward(x_batch)
            print("Loss: ", self.backward(x_batch, y_batch))
            print("W", self.weights)
            print("b", self.bias)
    
        
        
    
