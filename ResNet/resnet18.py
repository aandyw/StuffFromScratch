"""
Building an 18-layer residual network (ResNet-18) from scratch.
From the paper "Deep Residual Learning for Image Recognition" (https://arxiv.org/abs/1512.03385)
"""

import torchvision
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """The Residual Block"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False) -> None:
        """
        Create the Residual Block

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            stride (int): stride of first 3x3 convolution layer
            downsample (bool): whether to adjust for spatial dimensions due to downsampling via stride=2
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # For downsampling, the skip connection will pass through the 1x1 conv layer with stride of 2 to
        # match the spatial dimension of the downsampled feature maps and channels for the add operation.
        #
        # More specifically, the 'downsample block' is used for layer 2, 3, 4 of ResNet18 where the first conv2d
        # layer of the BasicBlock uses a stride of 2 instead of 1 to downsample feature maps for a larger
        # receptive field.
        # This is why we need to carefully craft our 'downsample block' to make sure spatial dimensions are
        # not disrupted when we add the skip connection in these residual blocks.
        self.downsample = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x.clone()
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.downsample:  # if layer not None
            identity = self.downsample(identity)

        x += identity
        o = self.relu(x)

        return o


class ResNet18(nn.Module):
    """The ResNet-18 Model"""

    def __init__(self, n_classes: int = 10) -> None:
        """
        Create the ResNet-18 Model

        Args:
            n_classes (int, optional): The number of output classes we predict for. Defaults to 10.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2, downsample=True),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2, downsample=True),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2, downsample=True),
            BasicBlock(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # our fully connected layer will be different to accomodate for CIFAR-10
        self.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # [bs, 512, 1, 1]

        x = torch.squeeze(x)  # reshape to [bs, 512]
        o = self.fc(x)

        return o

    @classmethod
    def from_pretrained(cls, model_type: str) -> nn.Module:
        """
        Load pretrained PyTorch ResNet-18 weights into our ResNet-18 implementation

        Inspired by Andrej Karpathy from 'Let's reproduce GPT-2 (124M)'
        (https://www.youtube.com/watch?v=l8pRSuU81PU)
        """

        assert model_type in {"resnet18"}, "only supports resnet18"
        print("loading weights from pytorch pretrained resnet18")

        # our model
        model = ResNet18(n_classes=10)
        r18 = model.state_dict()
        r18_keys = r18.keys()

        # pretrained pytorch resnet18 model
        p_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        p_r18 = p_model.state_dict()
        p_r18_keys = p_r18.keys()

        assert len(p_r18_keys) == len(r18_keys), f"mistmatched keys: {len(p_r18_keys)} != {len(r18_keys)}"
        # load weights from pretrained
        for k in p_r18_keys:
            if k.startswith("fc"):  # skip fc layer, we add our own for CIFAR-10
                continue

            assert p_r18[k].shape == r18[k].shape
            with torch.no_grad():
                r18[k].copy_(p_r18[k])

        return model
