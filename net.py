from mnist import MNIST

mndata = MNIST("samples")

images, labels = mndata.load_training()
images, labels = list(images), list(labels)

def display_image(image):
    for j in range(28):
        print([image[i] for i in range(j*28,(j+1)*28)])



import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) / (1 + np.exp(-x))**2



H_LAYER_SIZE = 8
H_LAYER_COUNT = 2

class neuron:
    def __init__(self):
        self.weights_in = []
        self.bias = None
        self.output = None



class network:

    def __init__(self):
        
        global neuron
        
        self.h_layers = [[neuron()]*H_LAYER_SIZE for i in range(H_LAYER_COUNT)]
        self.input_layer = [neuron() for i in range(784)]
        self.final_layer = [neuron()]*10
        
        r = np.random.default_rng()
        
        for neuron in self.h_layers[0]:
            neuron.weights_in = list(r.random((784,)))
            neuron.bias = r.random()
                
        for layer in self.h_layers[1:]:
            for neuron in layer:
                neuron.weights_in = list(r.random((H_LAYER_SIZE,)))
                neuron.bias = r.random()
            
        for neuron in self.final_layer:
            neuron.weights_in = list(r.random((H_LAYER_SIZE,)))
            neuron.bias = r.random()


    def load_image(self, images, index):
        for i in range(0,784):
            self.input_layer[i].output = images[index][i]/255


    def run_image(self):
        # Populate 1st hidden layer
        for i in range(0,H_LAYER_SIZE):
            self.h_layers[0][i].output = sigmoid(sum(
                [self.input_layer[j].output * self.h_layers[0][i].weights_in[j] for j in range(0,784)]
                ) + self.h_layers[0][i].bias)
            
        # Populate other hidden layers
        for i in range(1,H_LAYER_COUNT):
            for j in range(0,H_LAYER_SIZE):
                self.h_layers[i][j].output = sigmoid(sum(
                    [self.h_layers[i-1][k].output * self.h_layers[i][j].weights_in[k] for k in range(0,H_LAYER_SIZE)]
                    ) + self.h_layers[i][j].bias)
        
        # Populate final layer
        for i in range(0,10):
            self.final_layer[i].output = sigmoid(sum(
                [self.h_layers[H_LAYER_COUNT-1][j].output * self.final_layer[i].weights_in[j] for j in range(0,H_LAYER_SIZE)]
                ) + self.final_layer[i].bias)
            
        max = 0
        for i in range(0,10):
            if self.final_layer[i].output > self.final_layer[max].output:
                max = i
        
        # Return index of maximal final neuron output
        return max
    
if __name__ == "__main__":
    n = network()
    n.load_image(images, 0)
    print(n.run_image())