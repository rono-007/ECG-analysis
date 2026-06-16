import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # Added bias=True to match your 'Unexpected key' error log
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=True)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, stride=1, padding=3, bias=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # Added bias=True here as well
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)

class EcgModel(nn.Module):
    def __init__(self):
        super(EcgModel, self).__init__()
        # Removed the initial conv1/bn1 because your weights start directly at layer1
        # Layer 1 takes 12 channels (Leads) and outputs 24
        self.layer1 = ResidualBlock(12, 24, stride=2)
        self.layer2 = ResidualBlock(24, 32, stride=2)
        self.layer3 = ResidualBlock(32, 48, stride=2)
        self.layer4 = ResidualBlock(48, 64, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 5) 

    def forward(self, x):
        # Input shape: [Batch, 12, 5000]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x