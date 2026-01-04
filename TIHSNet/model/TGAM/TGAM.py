import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base import SegmentationHead
# from segmentation_models_pytorch.unet.decoder import UnetDecoder

class ThermalGradientAttention(nn.Module):

    def __init__(self, in_channels, reduction_ratio=16):
        super(ThermalGradientAttention, self).__init__()
        self.in_channels = in_channels

        # 确保通道数不会为零
        # 如果 in_channels 小于 reduction_ratio，则使用更小的reduction值
        self.reduction_ratio = min(reduction_ratio, max(1, in_channels // 2))
        reduced_channels = max(1, in_channels // self.reduction_ratio)
        
        # 梯度提取卷积
        self.grad_x = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.grad_y = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        
        # 初始化为Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        
        # 将Sobel算子复制到所有通道
        sobel_x = sobel_x.repeat(in_channels, 1, 1, 1)
        sobel_y = sobel_y.repeat(in_channels, 1, 1, 1)
        
        # 设置梯度算子权重（不参与训练）
        self.grad_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        self.grad_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        
        # 通道注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用安全的通道数
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        # 空间注意力
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        # 计算x和y方向的梯度
        grad_x = self.grad_x(x)
        grad_y = self.grad_y(x)
        
        # 梯度幅值
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 通道注意力 - 使用梯度信息
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(gradient_magnitude))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(gradient_magnitude))))
        channel_attention = self.sigmoid(avg_out + max_out)
        
        # 空间注意力
        avg_out = torch.mean(gradient_magnitude, dim=1, keepdim=True)
        max_out, _ = torch.max(gradient_magnitude, dim=1, keepdim=True)
        spatial_attention = torch.cat([avg_out, max_out], dim=1)
        spatial_attention = self.sigmoid(self.conv_spatial(spatial_attention))
        
        # 结合通道和空间注意力
        attention = channel_attention * spatial_attention
        
        # 应用注意力到原始输入
        return x * attention

class ThermalGradientUNet(nn.Module):

    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", classes=1, activation=None):
        super(ThermalGradientUNet, self).__init__()
        
        # 基本的U-Net模型
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=classes,
            activation=activation,
        )
        
        # 获取编码器
        self.encoder = self.base_model.encoder
        
        # 获取解码器
        self.decoder = self.base_model.decoder
        
        # 获取分割头
        self.segmentation_head = self.base_model.segmentation_head
        
        # 添加热梯度注意力模块（在编码器之前）
        self.thermal_attention = ThermalGradientAttention(in_channels=3)  # 对RGB输入应用
    
    def forward(self, x):
        # 应用热梯度注意力
        x_attended = self.thermal_attention(x)
        
        # 获取编码特征
        features = self.encoder(x_attended)
        
        # 解码
        decoder_output = self.decoder(*features)
        
        # 分割头
        masks = self.segmentation_head(decoder_output)
        
        return masks

# 使用示例
def create_thermal_gradient_unet(encoder_name="resnet34", classes=2, activation="sigmoid"):

    model = ThermalGradientUNet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        classes=classes,
        activation=activation,
    )
    return model

# 另一种实现方式：直接修改现有的U-Net模型
def add_thermal_attention_to_unet(unet_model):

    class UNetWithThermalAttention(nn.Module):
        def __init__(self, base_model):
            super(UNetWithThermalAttention, self).__init__()
            self.thermal_attention = ThermalGradientAttention(in_channels=3)
            self.base_model = base_model
        
        def forward(self, x):
            x_attended = self.thermal_attention(x)
            return self.base_model(x_attended)
    
    return UNetWithThermalAttention(unet_model)