import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        conv1 = nn.Conv2d(3, 6, 5)
        conv2 = nn.Conv2d(6, 16, 5)
        pool = nn.MaxPool2d(2, 2)

        fc1 = nn.Linear(16 * 5 * 5, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 10)

        self.conv_module = nn.Sequential(
            conv1, nn.ReLU(), pool,
            conv2, nn.ReLU(), pool
        )

        self.fc_module = nn.Sequential(
            fc1, nn.ReLU(),
            fc2, nn.ReLU(),
            fc3
        )

    def forward(self, x):
        x = self.conv_module(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc_module(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        # print(size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
