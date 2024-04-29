import snntorch as snn
import torch
from torchvision import datasets, transforms
from snntorch import utils
batch_size = 128
data_path = 'tmp/data/mnist'
num_classes = 10

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)

subset = 10
mnist_train = utils.data_subset(mnist_train, subset)

# Torch variables
dtype = torch.float

# practice encoding spikes from visual dataset

