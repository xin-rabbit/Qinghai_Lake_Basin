import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. 定义 CBAM 相关的注意力模块

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 避免通道数太少时 ratio 除法导致为 0
        inter_planes = max(in_planes // ratio, 1)
        
        self.fc1   = nn.Conv2d(in_planes, inter_planes, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2   = nn.Conv2d(inter_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)       # 通道注意力加权
        result = out * self.sa(out) # 空间注意力加权
        return result



# 2. 基础卷积块
def conv3x3_bn_relu(in_channels, out_channels, dropout=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    if dropout:
        layers.append(nn.Dropout2d(0.5))
    return nn.Sequential(*layers)



# 3. 集成 CBAM 与动态插值的 U-Net
class UNet_CBAM(nn.Module):
    def __init__(self, num_bands):
        super(UNet_CBAM, self).__init__()
        self.num_bands = num_bands
        
        # 注意：这里保留了 self.up 用于最后的输出层上采样，
        # 中间的跳跃连接上采样改用了更稳健的 F.interpolate
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Encoder 编码器
        self.down_conv1 = conv3x3_bn_relu(num_bands, 16, dropout=True)
        self.down_conv2 = conv3x3_bn_relu(16, 32, dropout=True)
        self.down_conv3 = conv3x3_bn_relu(32, 64, dropout=True)
        self.down_conv4 = conv3x3_bn_relu(64, 128, dropout=True)
        
        # Decoder 解码器及其对应的 CBAM 模块
        self.up_conv1 = conv3x3_bn_relu(192, 64, dropout=True)
        self.cbam1 = CBAM(64) 
        
        self.up_conv2 = conv3x3_bn_relu(96, 48, dropout=True)
        self.cbam2 = CBAM(48) 
        
        self.up_conv3 = conv3x3_bn_relu(64, 32)
        self.cbam3 = CBAM(32) 
        
        self.outp = nn.Sequential(
                nn.Conv2d(32, 1, kernel_size=3, padding=1),
                nn.Sigmoid()) 

    def forward(self, x):  
        ## ================= 编码器 (Encoder) =================
        conv1 = self.down_conv1(x)              
        x1 = F.avg_pool2d(input=conv1, kernel_size=2)  
        
        conv2 = self.down_conv2(x1)              
        x2 = F.avg_pool2d(input=conv2, kernel_size=2)  
        
        conv3 = self.down_conv3(x2)              
        x3 = F.avg_pool2d(input=conv3, kernel_size=2)  
        
        conv4 = self.down_conv4(x3)              
        x4 = F.avg_pool2d(input=conv4, kernel_size=2)  
        
        ## ================= 解码器 (Decoder) =================
        up_x4 = F.interpolate(x4, size=x3.shape[2:], mode='nearest')
        x4_up = torch.cat([up_x4, x3], dim=1)  
        
        x3_up = self.up_conv1(x4_up)  
        x3_up = self.cbam1(x3_up)     
        
        up_x3 = F.interpolate(x3_up, size=x2.shape[2:], mode='nearest')
        x3_up_cat = torch.cat([up_x3, x2], dim=1)  
        
        x2_up = self.up_conv2(x3_up_cat)  
        x2_up = self.cbam2(x2_up)         
        
        up_x2 = F.interpolate(x2_up, size=x1.shape[2:], mode='nearest')
        x2_up_cat = torch.cat([up_x2, x1], dim=1)  
        
        x1_up = self.up_conv3(x2_up_cat)  
        x1_up = self.cbam3(x1_up)         
        
        # 【最核心的修复】：最后一次上采样，直接强制对齐到最原始输入 x 的尺寸
        x1_final = F.interpolate(x1_up, size=x.shape[2:], mode='nearest')  
        logits = self.outp(x1_final)
        
        return logits