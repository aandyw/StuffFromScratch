"""Building an 18-layer residual network (ResNet-18) from scratch."""

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):

    def __init__(self):
        return

    def forward(self, x):
        return


class ResNet(nn.Module):
    def __init__(self):
        return

    def forward(self, x):
        return


def main():
    BATCH_SIZE = 4

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


if __name__ == "__main__":
    main()
