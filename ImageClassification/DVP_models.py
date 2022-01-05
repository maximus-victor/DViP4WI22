import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from config import config


class BaseNet(nn.Module):
  def __init__(self, out_classes, input_shape=(1, 256, 256)):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=config['kernel_size'])
    self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=config['kernel_size'])

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()

    n_size = self._get_conv_output(input_shape)

    self.fc1 = nn.Linear(in_features=n_size, out_features=config['fc_layer_size'])
    self.fc2 = nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes)

  def _get_conv_output(self, shape):
    batch_size = 1
    input = torch.autograd.Variable(torch.rand(batch_size, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size
  
  def _forward_features(self, x):
    x = self.pool1(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))

    return x
  
  def forward(self, x):
      # Here, we define how an input x is translated into an output. In our linear example, this was simply (x^T * w), now it becomes more complex but
      # we don't have to care about that (gradients etc. are taken care of by Pytorch).
      x = self._forward_features(x)
      # You can always print shapes and tensors here. This is very very helpful to debug.
      # print("x.shape:", x.shape)
      x = x.view(x.size(0), -1)
      x = F.relu(self.fc1(x))
      x = self.fc2(x)
      return x


class BaseNet8(nn.Module):
    def __init__(self, out_classes, input_shape=(1, 256, 256)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu7 = nn.ReLU()
        self.pool7 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool2d(kernel_size=2)


        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=n_size, out_features=config['fc_layer_size'])
        self.fc2 = nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))

        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.pool6(self.relu6(self.conv6(x)))
        x = self.pool7(self.relu7(self.conv7(x)))
        x = self.pool8(self.relu8(self.conv8(x)))

        return x

    def forward(self, x):
        # Here, we define how an input x is translated into an output. In our linear example, this was simply (x^T * w), now it becomes more complex but
        # we don't have to care about that (gradients etc. are taken care of by Pytorch).
        x = self._forward_features(x)
        # You can always print shapes and tensors here. This is very very helpful to debug.
        # print("x.shape:", x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaseNet8plus(nn.Module):
    def __init__(self, out_classes, input_shape=(1, 256, 256)):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=4)
        self.relu1 = nn.ReLU6()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)
        self.relu2 = nn.ReLU6()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.relu3 = nn.ReLU6()
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=32)
        self.relu4 = nn.ReLU6()
        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=64)
        self.relu5 = nn.ReLU6()
        self.pool5 = nn.MaxPool2d(kernel_size=2)

        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=128)
        self.relu6 = nn.ReLU6()
        self.pool6 = nn.MaxPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=256)
        self.relu7 = nn.ReLU6()
        self.pool7 = nn.MaxPool2d(kernel_size=2)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=512)
        self.relu8 = nn.ReLU6()
        self.pool8 = nn.MaxPool2d(kernel_size=2)


        n_size = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(in_features=n_size, out_features=config['fc_layer_size'])
        self.fc2 = nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes)

    def _get_conv_output(self, shape):
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(batch_size, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))

        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = self.pool8(self.relu8(self.bn8(self.conv8(x))))

        return x

    def forward(self, x):
        # Here, we define how an input x is translated into an output. In our linear example, this was simply (x^T * w), now it becomes more complex but
        # we don't have to care about that (gradients etc. are taken care of by Pytorch).
        x = self._forward_features(x)
        # You can always print shapes and tensors here. This is very very helpful to debug.
        # print("x.shape:", x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RegNet(nn.Module):
  def __init__(self, out_classes, input_shape=(1, 224, 224)):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=config['kernel_size'])
    self.batchnorm1 = nn.BatchNorm2d(32)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=config['kernel_size'])
    self.batchnorm2 = nn.BatchNorm2d(64)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=config['kernel_size'])
    self.batchnorm3 = nn.BatchNorm2d(128)

    self.pool1 = nn.MaxPool2d(kernel_size=2)
    self.pool2 = nn.MaxPool2d(kernel_size=2)
    self.pool3 = nn.MaxPool2d(kernel_size=2)

    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.relu3 = nn.ReLU()

    n_size = self._get_conv_output(input_shape)

    self.fc1 = nn.Linear(in_features=n_size, out_features=config['fc_layer_size'])
    self.relu2_1 = nn.ReLU()
    self.fc2 = nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes)

    self.softmax = nn.LogSoftmax(dim=1) 
    self.dropout = nn.Dropout(config['dropout'])

  def _get_conv_output(self, shape):
    batch_size = 1
    input = torch.autograd.Variable(torch.rand(batch_size, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size
  
  def _forward_features(self, x):
    x = self.poo1l(self.relu1(self.conv1(x)))
    x = self.pool2(self.relu2(self.conv2(x)))
    x = self.pool3(self.relu3(self.conv3(x)))
    return x
  
  def forward(self, x):
      x = self._forward_features(x)
      # print("x.shape:", x.shape)
      x = x.view(x.size(0), -1)
      x = self.relu2_1(self.fc1(x))
      x = self.dropout(x)
      x = self.fc2(x)
      x = self.softmax(x)
      return x


class resBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
      super().__init__()
      self.in_channels = in_channels
      self.out_channels = out_channels
      self.kernel_size = kernel_size
      self.stride = stride
      self.downsample = None

      if stride != 1 or in_channels != out_channels:
        self.downsample = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride),
                                        nn.BatchNorm2d(self.out_channels))

      self.block = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, padding=1),
                                nn.BatchNorm2d(self.out_channels),
                                nn.ReLU(),
                                nn.Conv2d(self.out_channels, self.out_channels, self.kernel_size, 1, padding=1),
                                nn.BatchNorm2d(self.out_channels))
      self.relu = nn.ReLU()

                               
    def forward(self, x):
      skip = x
      x = self.block(x)
      if self.downsample is not None:
        skip = self.downsample(skip)
      x += skip
      x = self.relu(x)
      return x


class ResiNet(nn.Module):
  def __init__(self, out_classes, input_shape=(1, 224, 224)):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=config['kernel_size'])
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=config['kernel_size'])
    self.max_pool = nn.MaxPool2d(kernel_size=2)
    self.dropout = nn.Dropout2d(config['dropout'])
    self.block1 = resBlock(in_channels=32, out_channels=32)
    self.block2 = resBlock(in_channels=32, out_channels=64, stride=2)
    self.block3 = resBlock(in_channels=64, out_channels=128, stride=2)
    self.block4 = resBlock(in_channels=128, out_channels=256)
    self.avg_pool = nn.AvgPool2d(kernel_size=2)

    n_size = self._get_conv_output(input_shape)

    self.fc1 = nn.Linear(in_features=n_size, out_features=config['fc_layer_size'])
    self.fc2 = nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes)

  def _get_conv_output(self, shape):
    batch_size = 1
    input = torch.autograd.Variable(torch.rand(batch_size, *shape))
    output_feat = self._forward_features(input)
    n_size = output_feat.data.view(batch_size, -1).size(1)
    return n_size
  
  def _forward_features(self, x):
    x = self.max_pool(F.relu(self.conv1(x)))
    x = self.max_pool(F.relu(self.conv2(x)))
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.avg_pool(x)
    return x

  def forward(self, x):
    x = self._forward_features(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ConnectionBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super().__init__()

        self.pathA = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=3,
                stride=stride,
                padding=1),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=3,
                stride=1,
                padding=1),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathB = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=5,
                stride=stride,
                padding=2),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=5,
                stride=1,
                padding=2),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathC = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=9,
                stride=stride,
                padding=4),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_c,
                out_channels=out_c,
                kernel_size=9,
                stride=1,
                padding=4),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c),
            nn.ReLU())

        self.pathSkip = nn.Sequential(
            nn.Conv2d(
                in_channels=in_c,
                out_channels=out_c,
                kernel_size=1,
                stride=stride),
            nn.Dropout2d(config['dropout']),
            nn.BatchNorm2d(num_features=out_c))

    def forward(self, x):
        return self.pathA(x) + self.pathB(x) + self.pathC(x) + self.pathSkip(x)


class IncNet(nn.Module):
    def __init__(self, out_classes, input_shape=(1, 224, 224)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConnectionBlock(in_c=64, out_c=64, stride=1),
            ConnectionBlock(in_c=64, out_c=128, stride=2),
            ConnectionBlock(in_c=128, out_c=256, stride=2),
            ConnectionBlock(in_c=256, out_c=512, stride=2),
            nn.AvgPool2d(kernel_size=(2, 2)),
            Flatten(),
            nn.Linear(in_features=4608, out_features=config['fc_layer_size']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(in_features=config['fc_layer_size'], out_features=config['fc_layer_size']),
            nn.ReLU(),
            nn.Linear(in_features=config['fc_layer_size'], out_features=out_classes),
            #nn.Sigmoid()
            )

    def forward(self, x):
      return self.net(x)


class OctaNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=16)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(num_features=128)

        self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(num_features=256)

        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(num_features=512)

        self.conv8 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(num_features=1024)

        self.relu1 = nn.ReLU6()
        self.relu2 = nn.ReLU6()
        self.relu3 = nn.ReLU6()
        self.relu4 = nn.ReLU6()
        self.relu5 = nn.ReLU6()
        self.relu6 = nn.ReLU6()
        self.relu7 = nn.ReLU6()
        self.relu8 = nn.ReLU6()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.pool6 = nn.MaxPool2d(kernel_size=2)
        self.pool7 = nn.MaxPool2d(kernel_size=2)
        self.pool8 = nn.MaxPool2d(kernel_size=2)


        self.fc1 = nn.Linear(in_features=1024 * 1 * 1, out_features=150)
        self.fc2 = nn.Linear(in_features=150, out_features=n_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        x = self.pool7(self.relu7(self.bn7(self.conv7(x))))
        x = self.pool8(self.relu8(self.bn8(self.conv8(x))))
        x = x.view(x.shape[0], 1024 * 1 * 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
