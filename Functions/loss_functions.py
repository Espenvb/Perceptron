import numpy as np
    
class MSE():
    def __init__(self):
        super().__init__()
        self.name = "MSE"
                
    def function(y, y_pred):
        return np.mean(y - y_pred)**2
       
    def derivative(y, y_pred):
        return -2*np.mean(y-y_pred)
    
class SE():
    def __init__(self):
        super().__init__()
        self.name = "MSE"
                
    def function(y, y_pred):
        return np.sum((y - y_pred)**2)
       
    def derivative(y, y_pred):
        return -2*np.sum(y-y_pred)
    

    