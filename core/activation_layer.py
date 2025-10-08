import numpy

from utils import activtations
from layers import Layer

class ActivationLayer(Layer):
    """
    Activation function layer
    """
    def __init__(self, name):
        super.__init__()
        self.func, self.derivative = activtations.get_activations(name)
        self.input = None

        def forward(self, x):
            self.input = x
            return self.function(x)
        
        def backward(self, grad_output):
            grad_input =  grad_output * self.derivative(self.input)
            return grad_input