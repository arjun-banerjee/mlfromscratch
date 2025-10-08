import numpy as np

from core.activation_layer import ActivationLayer
from layers import Layer

class SingleNeuron(Layer): 
    """
    Implements w^tx + b
    """

    def __init__(self, input_size):
        super().__init__()
        self.input = None
        #TODO: IMPLEMENT SMART INTIALIZERS
        self.params["W"] = np.random.rand((1, input_size))
        self.params["b"] = np.random.rand()
        self.grads = {}
    
    def forward(self, x):
        self.input = x
        return np.dot(self.params["W"], x) + self.params["b"]

    def backward(self, grad_output):
        #Derivative of d/dw (w^t x + b) = x
        #Derivative of d/db (w^t x + b) = 1
        #Derivative of d/dx (w^t x + b) = w^t
        self.grads["W"] = np.dot(self.input, grad_output)
        self.grads["b"] = self.input
        return np.dot(grad_output, self.params["W"])

