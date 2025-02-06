import numpy as np
    
class ReLu():
    def __init__(self):
        self.name = "ReLu Activation Function"
    def function(x):
        return np.maximum(x, 0)
    
    def derivative(x):
        return np.where(x>0, 1, 0)
    
class Heaviside():
    def __init__(self):
        super().__init__()
        self.name = "Heaviside"
                
    def function(y):
        output = np.array([])
        for i in y:
            if i > 0:
                output = np.append(output, 1)
            else:
                output = np.append(output, 0)
        return output
    
    def derivative(x):
        return 1