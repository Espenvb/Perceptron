import numpy as np

def l1(theta, l):
    return l * np.sum(np.linalg.norm(theta))

def l2(theta, l):
    return l * np.sum(np.linalg.norm(theta)**2)