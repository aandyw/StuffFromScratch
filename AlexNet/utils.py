import torch
import matplotlib.pyplot as plt
import numpy as np


def get_device():
    """Get available device"""

    if torch.cuda.is_available():
        print("Using CUDA...")
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using MPS...")
        return torch.device("mps")
    else:
        print("Using CPU...")
        return torch.device("cpu")


def imshow(img):
    """Display image"""

    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
