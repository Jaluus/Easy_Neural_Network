from pickle import load
import numpy as np
from src.network import Network, load_network
import src.mnist_loader as mnist_loader
import matplotlib.pyplot as plt

# load the images, with a different shape
training_data, validation_data, test_data = mnist_loader.load_data("data/mnist.pkl.gz")

# load the data
print(f"{len(training_data[0])} training images loaded")
print(f"{len(validation_data[0])} validation images loaded")
print(f"{len(test_data[0])} test images loaded")

# load the weights from a saved json file
net = load_network("mynet.json")

# plot a lot of images
def plot_image(training_data):
    fig, axes = plt.subplots(5, 10, figsize=(15, 7.5))

    for idx, ax in enumerate(axes.flat):

        eval_img = training_data[0][idx]
        detected_number, endlayer = net.evaluate_image(eval_img)
        ground_truth = training_data[1][idx]

        certainty = np.max(endlayer) / np.sum(endlayer)

        cmap = "Oranges"
        if ground_truth == detected_number:
            cmap = "Greens"

        ax.imshow(eval_img.reshape((28, 28)), cmap=cmap)
        ax.set(
            xticks=[],
            yticks=[],
            title=f"{detected_number} , {certainty * 100:.2f}%",
        )
    plt.show()


plot_image(training_data)
