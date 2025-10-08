import numpy as np

class Layer: 
    """Base class for all layers, including activation layers"""
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, x):
        raise NotImplementedError