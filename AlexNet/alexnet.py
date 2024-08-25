"""
Implementing the famous AlexNet Architecture from scratch from the 2012 paper
"ImageNet Classification with Deep Convolutional Neural Networks"
"""

import torch
import torch.nn as nn


class AlexNet(nn.Module):

    def __init__(self, n_classes: int = 1000) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)  # [bs, 256, 6, 6]
        x = x.reshape(x.size(0), -1)  # reshape to [bs, 6*6*256] = [bs, 9216]
        o = self.classifier(x)

        return o


# test basic forward pass of AlexNet
if __name__ == "__main__":
    alexnet = AlexNet()
    n_params = sum(p.numel() for p in alexnet.parameters())
    n_trainable_params = sum(p.numel() for p in alexnet.parameters() if p.requires_grad)

    print(f"Number of parameters: {n_params}")
    print(f"Number of trainable parameters: {n_trainable_params}")

    # paper mentions 224 x 224 but seems to be a mistake?
    dummy_image = torch.randn(1, 3, 227, 227)
    out = alexnet(dummy_image)

    assert out.shape == (1, 1000), f"Expected shape: (1, 1000) | Actual shape: {out.shape}"

    print(f"\nModel Summary:\n========\n{alexnet}")
