import numpy as np

class Activation_Function:
    def __init__(self):
        self.name = "Activation Function Base Class"
        
    def function(x):
        raise NotImplementedError
    
    def derivative(x):
        raise 
    
class ReLu(Activation_Function):
    def __init__(self):
        self.name = "ReLu Activation Function"
    @staticmethod
    def function(x):
        return np.maximum(x, 0)
    
    @staticmethod
    def derivative(x):
        return np.where(x>0, 1, 0)