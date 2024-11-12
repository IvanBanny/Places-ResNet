import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        residual = self.skip_connection(x)
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += residual
        out = nn.functional.relu(out)
        return out


class MyModel(nn.Module):
    def __init__(self, num_classes=100, dropout_rate=0.2):
        super(MyModel, self).__init__()
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout1 = nn.Dropout2d(p=self.dropout_rate)

        # Increase the number of residual blocks
        self.block1 = self._resnet_layers(64, 128, num_blocks=4)
        self.block2 = self._resnet_layers(128, 256, num_blocks=4)
        self.block3 = self._resnet_layers(256, 512, num_blocks=4)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

        # Reduce the size of the fully connected layer
        self.fc = nn.Linear(512, num_classes)

        self.features = nn.Sequential(
            self.conv1,
            self.bn1,
            nn.ReLU(),
            self.pool1,
            self.dropout1,
            self.block1,
            self.block2,
            self.block3,
            self.global_avg_pool,
            self.dropout2
        )

    @staticmethod
    def _resnet_layers(in_channels, out_channels, num_blocks):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, dropout_rate=0.2),
            *[ResidualBlock(out_channels, out_channels, dropout_rate=0.2) for _ in range(num_blocks)]
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
