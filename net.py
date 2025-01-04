import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

H_LAYER_SIZE = 8
H_LAYER_COUNT = 2

class neuron:
    def __init__(self):
        self.weights_in = [None]*H_LAYER_SIZE
        self.bias = None
        self.output = None
    
h_layers = [[neuron()]*H_LAYER_SIZE for i in range(H_LAYER_COUNT)]