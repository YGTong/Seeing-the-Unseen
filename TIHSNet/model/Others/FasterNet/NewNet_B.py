# from newnet.FasterNet.detection.backbones.fasternet import fasternet_s
from .detection.backbones.fasternet import fasternet_s
import torch
from torch import nn
from model.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)


class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.up(self.conv(x))


class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvDown, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(self.down(x))


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, out_channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activation(y)


class NewUnet3(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=True, backbone='vgg', *args, **kwargs):
        super(NewUnet3, self).__init__()
        self.backbone = fasternet_s(**kwargs)
        out_filters = [128, 256, 512, 1024]
        self.ridge = nn.Sequential(
            nn.Conv2d(out_filters[3], out_filters[3], 3, 1, 1),
            nn.BatchNorm2d(out_filters[3]),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[3], out_filters[2], 3, 1, 1),
            nn.BatchNorm2d(out_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.up1 = ConvUp(out_filters[3], out_filters[1])
        self.up2 = ConvUp(out_filters[2], out_filters[0])
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_filters[1], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.attention = SimAM()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_filters[0], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )

        self.down1 = ConvDown(out_filters[1], out_filters[1])
        self.down2 = ConvDown(out_filters[2], out_filters[2])
        self.down3 = ConvDown(out_filters[3], out_filters[3])

        self.short1 = nn.Sequential(
            nn.Conv2d(out_filters[1], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True)
        )
        self.short2 = nn.Sequential(
            nn.Conv2d(out_filters[2], out_filters[1], 3, 1, 1),
            nn.BatchNorm2d(out_filters[1]),
            nn.ReLU(inplace=True)
        )
        self.short3 = nn.Sequential(
            nn.Conv2d(out_filters[3], out_filters[2], 3, 1, 1),
            nn.BatchNorm2d(out_filters[2]),
            nn.ReLU(inplace=True)
        )
        self.segmentation_head = SegmentationHead(
            in_channels=1024,
            out_channels=2,
            activation=None,  # 输出为logits，如果需要概率可以选择'softmax'或'sigmoid'
            kernel_size=1,
            upsampling=32,
            dropout_rate=0.1,
        )

    def forward(self, x):
        [stage1, stage2, stage3, stage4] = self.backbone(x)
        out1 = torch.concat((stage3, self.ridge(stage4)), dim=1)
        out2 = torch.concat((stage2, self.up1(out1)), dim=1)
        out3 = torch.concat((stage1, self.up2(out2)), dim=1)
        out = self.conv1(out3)
        out = self.conv2(self.attention(out))
        out4 = torch.concat((out, self.short1(out3)), dim=1)
        out5 = torch.concat((self.short2(out2), self.down1(out4)), dim=1)
        out6 = torch.concat((self.short3(out1), self.down2(out5)), dim=1)
        out7 = self.down3(out6)
        out7 = self.segmentation_head(out7)
        return out7

if __name__ == '__main__':
    net = NewUnet3(3, 1, False)
    inputs = torch.randn((1, 3, 512, 512))
    outputs = net(inputs)
    print(net)

