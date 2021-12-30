import numpy as np
import torch
from torch.nn import Module, Conv2d, MaxPool2d, Linear
from torch.nn.functional import relu


class TorchConvNet(Module):
    # TODO: Should the kernel size and stride get bigger or smaller the deeper in the network?
    def __init__(self, n_output):
        super(TorchConvNet, self).__init__()
        self.conv_input = Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), stride=2)
        self.pool2 = MaxPool2d(kernel_size=(2, 2))
        self.conv_64 = Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=2)
        self.pool4 = MaxPool2d(kernel_size=(4, 4))
        self.conv_128 = Conv2d(in_channels=64, out_channels=128, kernel_size=(7, 7), stride=3)

        # Hack to get dimensions of first fully connected layer:
        x_ = torch.randn(3, 1920, 1920).view(-1, 3, 1920, 1920)
        self._first_linear_layer_shape = None
        self.convs(x_)

        self.fc1 = Linear(in_features=self._first_linear_layer_shape, out_features=256)
        self.fc2 = Linear(in_features=256, out_features=n_output)

    # This method expects tensors of shape (num_examples, width, height, num_channels)
    def convs(self, x):
        x = self.pool2(relu(self.conv_input(x)))
        x = self.pool4(relu(self.conv_64(x)))
        x = self.pool4(relu(self.conv_128(x)))

        if self._first_linear_layer_shape is None:
            self._first_linear_layer_shape = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._first_linear_layer_shape)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        return x
