import torch
import torch.nn as nn
import torch.nn.functional as f

class CNN (nn.Module):
    def __init__(self, feature, num_class):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, feature, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(feature)
        self.conv2 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(feature)
        self.conv3 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(feature)
        self.conv4 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(feature)
        self.conv5 = nn.Conv2d(feature, feature, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(feature)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(feature*16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_class)
    def forward(self, x):

        #224
        out = self.conv1(x)
        out= self.bn1(out)
        out = self.relu(out)
        out = self.pool(out)
        #56
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool(out)
        #56
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool(out)
        #28
        out=self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.pool(out)
        #14
        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.pool(out)
        #7
        out = self.fc1(out.view(out.size(0), -1))
        out = self.fc2(out)
        return out




class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=124):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x