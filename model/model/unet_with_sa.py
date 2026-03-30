## des: a simple U-Net model with CBAM attention

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------- CBAM Module -------------------
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ------------------- U-Net with CBAM -------------------
def conv3x3_bn_relu(in_channels, out_channels, dropout=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)


class unet(nn.Module):
    def __init__(self, num_bands):
        super(unet, self).__init__()
        self.num_bands = num_bands
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.down_conv1 = conv3x3_bn_relu(num_bands, 16, dropout=True)
        self.down_conv2 = conv3x3_bn_relu(16, 32, dropout=True)
        self.down_conv3 = conv3x3_bn_relu(32, 64, dropout=True)
        self.down_conv4 = conv3x3_bn_relu(64, 128, dropout=True)

        # CBAM applied on bottleneck features (128 channels)
        self.cbam = CBAM(in_channels=128, reduction=16, spatial_kernel=7)

        self.up_conv1 = conv3x3_bn_relu(192, 64, dropout=True)   # 128+64=192
        self.up_conv2 = conv3x3_bn_relu(96, 48, dropout=True)    # 64+32=96
        self.up_conv3 = conv3x3_bn_relu(64, 32)                  # 48+16=64
        self.outp = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):  ## input size: 6x256x256 (num_bands x H x W)
        ## encoder part
        x1 = self.down_conv1(x)               # 16x256x256
        x1 = F.avg_pool2d(input=x1, kernel_size=2)  # 16x128x128

        x2 = self.down_conv2(x1)              # 32x128x128
        x2 = F.avg_pool2d(input=x2, kernel_size=2)  # 32x64x64

        x3 = self.down_conv3(x2)              # 64x64x64
        x3 = F.avg_pool2d(input=x3, kernel_size=2)  # 64x32x32

        x4 = self.down_conv4(x3)              # 128x32x32
        x4 = F.avg_pool2d(input=x4, kernel_size=2)  # 128x16x16

        # Apply CBAM attention on bottleneck features
        x4 = self.cbam(x4)                    # 128x16x16 (enhanced)

        ## decoder part
        x4_up = torch.cat([self.up(x4), x3], dim=1)  # (128+64)x32x32
        x3_up = self.up_conv1(x4_up)                 # 64x32x32
        x3_up = torch.cat([self.up(x3_up), x2], dim=1)  # (64+32)x64x64
        x2_up = self.up_conv2(x3_up)                 # 48x64x64
        x2_up = torch.cat([self.up(x2_up), x1], dim=1)  # (48+16)x128x128
        x1_up = self.up_conv3(x2_up)                 # 32x128x128
        x1_up = self.up(x1_up)                       # 32x256x256
        logits = self.outp(x1_up)                    # 1x256x256
        return logits