import numpy as np
"""
A file containing various activation functions useful in modern ML, and their derivatives (for backprop purposes).
"""
def get_activations(name: str):
    #TODO
    pass

def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Implements the sigmoid: 1/(1+e^-x)
    """
    return np.reciprocal(1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """
    Implements the sigmoid derivative. 
    (1 + e^(-x))^2 * e^-x =  sigmoid * (1 - sigmoid)
    """
    sigmoid_computation = sigmoid(x)
    return np.multiply(sigmoid_computation, np.subtract(1, sigmoid_computation))

def tanh(x: np.ndarray) -> np.ndarray:
    #TODO
    pass

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    #TODO
    pass


def relu(x: np.ndarray) -> np.ndarray:
    """
    Implements RELU, returning 0 if the element is less than 0 and the element otherwise
    """
    return np.maximum(x, 0)

def relu_gradient(x: np.ndarray) -> np.ndarray:
    """
    Implements the gradient of RELU, returning 0 if the element is less than 0 and 1 otherwise. Assumes 0 for the nondifferntiable corner
    """
    return (x>0).astype(float)

def leaky_relu(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Implements the leaky relu: alpha*x if x < 0, else x
    """
    mask = (x<0)
    return x*alpha*mask + x*(~mask)

def leaky_relu_derivative(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Implements the leaky relu derivative: alpha if x < 0, else x
    """
    mask = (x<0)
    return alpha*mask + (~mask)


#TODO: PARAMETRIC RELU

def elu(x: np.ndarray) -> np.ndarray:
    #TODO
    pass

def elu_derivative(x: np.ndarray) -> np.ndarray:
    #TODO
    pass


def selu(x: np.ndarray) -> np.ndarray:
    #TODO: Used with LeCun init + AlphaDropout
    pass

def selu_derivative(x: np.ndarray) -> np.ndarray:
    #TODO: Used with LeCun init + AlphaDropout
    pass

def gelu(x: np.ndarray) -> np.ndarray:
    #TODO
    pass

def gelu_derivative(x: np.ndarray) -> np.ndarray:
    #TODO
    pass

def silu(x: np.ndarray) -> np.ndarray:
    #TODO
    pass

def silu_derivative(x: np.ndarray) -> np.ndarray:
    #TODO
    pass



