"""2D UNet for slice-by-slice segmentation.

Same architecture as Ais cryoPom-bxe / cryoPom-dice models:
4 encoder blocks, bottleneck, 4 decoder blocks with skip connections.
Dropout after encoder blocks 3/4 and in bottleneck + decoder blocks 1-3.
"""

import torch
import torch.nn as nn


class _EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x, skip


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv1 = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class UNet2D(nn.Module):
    """2D UNet matching the Ais cryoPom architecture.

    Encoder:  64 -> 128 -> 256 -> 512
    Bottleneck: 1024
    Decoder:  512 -> 256 -> 128 -> 64
    Output:   1 channel, sigmoid
    """

    def __init__(self):
        super().__init__()
        drop = 0.15
        drop_bn = 0.3

        # Encoder
        self.enc1 = _EncoderBlock(1, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256, dropout=drop)
        self.enc4 = _EncoderBlock(256, 512, dropout=drop)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(1024)
        self.bottleneck_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bottleneck_bn2 = nn.BatchNorm2d(1024)
        self.bottleneck_drop = nn.Dropout2d(drop_bn)

        # Decoder
        self.dec1 = _DecoderBlock(1024, 512, 512, dropout=drop)
        self.dec2 = _DecoderBlock(512, 256, 256, dropout=drop)
        self.dec3 = _DecoderBlock(256, 128, 128, dropout=drop)
        self.dec4 = _DecoderBlock(128, 64, 64)

        # Output
        self.out_conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        # Encoder
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)

        # Bottleneck
        x = torch.relu(self.bottleneck_bn1(self.bottleneck_conv1(x)))
        x = torch.relu(self.bottleneck_bn2(self.bottleneck_conv2(x)))
        x = self.bottleneck_drop(x)

        # Decoder
        x = self.dec1(x, skip4)
        x = self.dec2(x, skip3)
        x = self.dec3(x, skip2)
        x = self.dec4(x, skip1)

        # Output
        x = torch.sigmoid(self.out_conv(x))
        return x
