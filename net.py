from mnist import MNIST

mndata = MNIST("samples")

images, labels = mndata.load_training()
images, labels = list(images), list(labels)



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
    
h_layers = [[neuron()]*H_LAYER_SIZE for i in range(H_LAYER_COUNT)]

input_layer = [neuron() for i in range(784)]

final_layer = [neuron()]*10

# print(mndata.display(images[0]))

def randomize_weights_and_biases():
    r = np.random.default_rng()
    
    for neuron in h_layers[0]:
        neuron.weights_in = r.random(784,)
        neuron.bias = r.random()
            
    for layer in h_layers[1:]:
        for neuron in layer:
            neuron.weights_in = r.random(H_LAYER_SIZE,)
            neuron.bias = r.random()
        
    for neuron in final_layer:
        neuron.weights_in = r.random(H_LAYER_SIZE,)
        neuron.bias = r.random()


def run_one_image(images, index):
    for i in range(0,784):
        input_layer[i].output = sigmoid(images[index][i])
        
    # Populate 1st hidden layer
    for i in range(0,H_LAYER_SIZE):
        h_layers[0][i].output = sigmoid(sum(
            [input_layer[j].output * h_layers[0][i].weights_in[j] for j in range(0,784)]
            ) + h_layers[0][i].bias)
        
    # Populate other hidden layers
    for i in range(1,H_LAYER_COUNT):
        for j in range(0,H_LAYER_SIZE):
            h_layers[i][j].output = sigmoid(sum(
                [h_layers[i-1][k].output * h_layers[i][j].weights_in[k] for k in range(0,H_LAYER_SIZE)]
                ) + h_layers[i][j].bias)
    
    # Populate final layer
    for i in range(0,10):
        final_layer[i].output = sigmoid(sum(
            [h_layers[H_LAYER_COUNT-1][j].output * final_layer[i].weights_in[j] for j in range(0,H_LAYER_SIZE)]
            ) + final_layer[i].bias)
        
    print(list(enumerate([(j,i) for (i,j) in enumerate([final_layer[i].output for i in range(0,10)])])))
    print(list(enumerate([final_layer[i].output for i in range(0,10)])))
    # Return index of maximal final neuron output
    return max(enumerate([(j,i) for (i,j) in enumerate([final_layer[i].output for i in range(0,10)])]))[0]
# TODO: read pixel number scaling 0-255 docs