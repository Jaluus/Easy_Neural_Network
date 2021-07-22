import numpy as np
from src.network import Network
import src.mnist_loader as mnist_loader
import matplotlib.pyplot as plt

# Load data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper(
    "data/mnist.pkl.gz"
)

# Just print out the size of the training data
print(f"{len(training_data)} training images loaded")
print(f"{len(validation_data)} validation images loaded")
print(f"{len(test_data)} test images loaded")

# a few Hyperparameters
learning_rate = 3.0
epochs = 5
mini_batch_size = 10

# generate a neural net with 784 input nodes, 30 hidden nodes, and 10 output nodes
net = Network([784, 30, 10])

# train the neural net with stochastic grad descent
net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)

# save the weights of the network
net.save("mynet.json")
