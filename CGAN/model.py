import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms


class TwoConvBlock_2D(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding="same")
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.rl = nn.LeakyReLU()
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding="same")
    self.bn2 = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.rl(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.rl(x)
    return x

class Discriminator(nn.Module):   #識別器
  def __init__(self):
    super().__init__()
    self.conv1 = TwoConvBlock_2D(1,64)
    self.conv2 = TwoConvBlock_2D(64, 128)
    self.conv3 = TwoConvBlock_2D(128, 256)

    self.avgpool_2D = nn.AvgPool2d(2, stride = 2)
    self.global_avgpool_2D = nn.AvgPool2d(7)

    self.l1 = nn.Linear(256, 20)
    self.l2 = nn.Linear(20, 1)
    self.rl = nn.LeakyReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.1)
    self.embed = nn.Embedding(10, 256)

  def forward(self, x, y):
    x = self.conv1(x)
    x = self.avgpool_2D(x)
    x = self.conv2(x)
    x = self.avgpool_2D(x)
    x = self.conv3(x)
    x = self.global_avgpool_2D(x)
    x = x.view(-1, 256)
    _x = x
    x = self.dropout1(x)
    x = self.l1(x)
    x = self.rl(x)
    x = self.dropout2(x)
    x = self.l2(x)

    _y = self.embed(y)   #ラベルをembedding層で埋め込む
    xy = (_x*_y).sum(1, keepdim=True)   #出力ベクトルとの内積をとる

    x = x+xy   #内積を加算する
    x = torch.sigmoid(x)
    return x

class Generator(nn.Module):   #生成器
  def __init__(self):
    super().__init__()
    self.l = nn.Linear(110, 49)
    self.dropout = nn.Dropout(0.2)
    self.TCB1 = TwoConvBlock_2D(1,512)
    self.TCB2 = TwoConvBlock_2D(512,256)
    self.TCB3 = TwoConvBlock_2D(256,128)
    self.UC1 = nn.ConvTranspose2d(512, 512, kernel_size =2, stride = 2)
    self.UC2 = nn.ConvTranspose2d(256, 256, kernel_size =2, stride = 2)
    self.conv1 = nn.Conv2d(128, 1, kernel_size = 2, padding="same")
  

  def forward(self, x, y):
    y = torch.nn.functional.one_hot(y.long(), num_classes=10).to(torch.float32)
    x = torch.cat([x, y], dim= 1)
    x = self.dropout(x)
    x = self.l(x)
    x = torch.reshape(x, (-1, 1, 7, 7))
    x = self.TCB1(x)
    x = self.UC1(x)
    x = self.TCB2(x)
    x = self.UC2(x)
    x = self.TCB3(x)
    x = self.conv1(x)
    x = torch.sigmoid(x)
    return x
