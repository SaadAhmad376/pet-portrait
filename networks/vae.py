import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  # Residual connection
        return F.relu(out)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1) # 32x256x256
        self.bn1 = nn.BatchNorm2d(32)
        self.resblock1 = ResidualBlock(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # 64x128x128
        self.bn2 = nn.BatchNorm2d(64)
        self.resblock2 = ResidualBlock(64)
        self.fc = nn.Linear(64 * 128 * 128, 20) 

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock2(x)
        x = x.view(-1, 64 * 128 * 128)
        return self.fc(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(20, 64 * 128 * 128)
        self.resblock1 = ResidualBlock(64)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.resblock2 = ResidualBlock(32)
        self.conv1 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, 64, 128, 128)
        z = self.resblock1(z)
        z = F.relu(self.bn2(self.conv2(z)))
        z = self.resblock2(z)
        return torch.sigmoid(self.conv1(z))

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

