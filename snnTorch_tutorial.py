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

def forward_euler_lif(pot, resist, cap, current, time_step):
    time_constant = resist * cap
    pot = pot + time_step * (current * resist - pot) / time_constant
    return pot
