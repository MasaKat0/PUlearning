import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class NN_dr(nn.Module):
    def __init__(self, dim):
        super(NN_dr, self).__init__()
        self.fc1=nn.Linear(dim, 100)
        self.fc2=nn.Linear(100, 1)

    def __call__(self, x):
        h = self.fc1(x)
        h = F.relu(h)
        h = self.fc2(h)
        return h

class CNN_dr(nn.Module):
    def __init__(self, dim):
        super(CNN_dr, self).__init__()
        self.dim = dim
        self.conv1=nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1)
        self.conv2=nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1)
        self.conv3=nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=2)
        self.conv4=nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1)
        self.conv5=nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv6=nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1, stride=2)
        self.conv7=nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, padding=1)
        self.conv8=nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1)
        self.conv9=nn.Conv2d(in_channels=192, out_channels=10, kernel_size=1)
        self.b1=nn.BatchNorm2d(96)
        self.b2=nn.BatchNorm2d(96)
        self.b3=nn.BatchNorm2d(96)
        self.b4=nn.BatchNorm2d(192)
        self.b5=nn.BatchNorm2d(192)
        self.b6=nn.BatchNorm2d(192)
        self.b7=nn.BatchNorm2d(192)
        self.b8=nn.BatchNorm2d(192)
        self.b9=nn.BatchNorm2d(10)
        self.fc1=nn.Linear(10 * self.dim * self.dim, 1000)
        self.fc2=nn.Linear(1000, 1000)
        self.fc3=nn.Linear(1000, 1)
        self.af = F.relu

    def __call__(self, x):
        h = self.conv1(x)
        h = self.b1(h)
        h = self.af(h)
        h = self.conv2(h)
        h = self.b2(h)
        h = self.af(h)
        h = self.conv3(h)
        h = self.b3(h)
        h = self.af(h)
        h = self.conv4(h)
        h = self.b4(h)
        h = self.af(h)
        h = self.conv5(h)
        h = self.b5(h)
        h = self.af(h)
        h = self.conv6(h)
        h = self.b6(h)
        h = self.af(h)
        h = self.conv7(h)
        h = self.b7(h)
        h = self.af(h)
        h = self.conv8(h)
        h = self.b8(h)
        h = self.af(h)
        h = self.conv9(h)
        h = self.b9(h)
        h = self.af(h)
        h = h.view(-1, 10 * self.dim * self.dim)
        h = self.fc1(h)
        h = self.af(h)
        h = self.fc2(h)
        h = self.af(h)
        h = self.fc3(h)
        return h
