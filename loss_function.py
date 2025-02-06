import numpy as np

class Loss_Function:
    def __init__(self):
        self.name = "Loss Function Base Class"
    @staticmethod   
    def function(x):
        raise NotImplementedError
    
    @staticmethod
    def derivative(x):
        raise 
    
class MSE(Loss_Function):
    def __init__(self):
        super().__init__()
        self.name = "MSE"
        
    @staticmethod          
    def function(y, y_pred):
        return np.mean(y - y_pred)**2
    
    @staticmethod      
    def derivative(y, y_pred):
        return 2*np.mean(y-y_pred)