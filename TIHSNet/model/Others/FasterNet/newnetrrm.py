from exmodel.FasterNet.detection.backbones.fasternet import fasternet_s
import torch
from torch import nn


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


class ConvUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
        )

    def forward(self, x):
        out1 = self.conv(x)
        out2 = self.up(out1)
        return out1, out2


class RefDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefDown, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.down(x)
        out2 = self.pool(out1)
        return out1, out2


class RefUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefUp, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels // 2, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(self.conv(x))


class RefNet(nn.Module):
    def __init__(self, num_classes):
        super(RefNet, self).__init__()
        out_filters = [128, 256, 512, 1024]
        self.conv = nn.Sequential(
            nn.Conv2d(out_filters[0], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True)
        )
        self.down1 = RefDown(out_filters[1], out_filters[1])
        self.down2 = RefDown(out_filters[2], out_filters[2])
        self.down3 = RefDown(out_filters[3], out_filters[3])
        self.up1 = RefUp(out_filters[3] * 2, out_filters[2])
        self.up2 = RefUp(out_filters[3], out_filters[1])
        self.up3 = RefUp(out_filters[2], out_filters[0])
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_filters[1], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Sequential(
            nn.Conv2d(out_filters[1], out_filters[0] // 2, 3, 1, 1),
            nn.BatchNorm2d(out_filters[0] // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_filters[0] // 2, num_classes, 3, 1, 1),
        )

    def forward(self, x, x1, x2, x3, x4):
        out = self.conv(x)
        out1_1, out1_2 = self.down1(torch.concat((out, x1), dim=1))
        out2_1, out2_2 = self.down2(torch.concat((out1_2, x2), dim=1))
        out3_1, out3_2 = self.down3(torch.concat((out2_2, x3), dim=1))
        out4 = torch.concat((x4, out3_2), dim=1)
        out5 = torch.concat((self.up1(out4), out3_1), dim=1)
        out6 = torch.concat((self.up2(out5), out2_1), dim=1)
        out7 = torch.concat((self.up3(out6), out1_1), dim=1)
        out = self.final(torch.concat((out, self.conv2(out7)), dim=1))
        return nn.UpsamplingBilinear2d(scale_factor=4)(out)


class newnetrrm(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, pretrained=True, backbone='vgg', *args, **kwargs):
        super(newnetrrm, self).__init__()

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
        )
        self.attention = SimAM()
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_filters[0], out_filters[0], 3, 1, 1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(inplace=True),
        )
        self.refine = RefNet(num_classes)

    def forward(self, x):
        [stage1, stage2, stage3, stage4] = self.backbone(x)
        out = torch.concat((stage3, self.ridge(stage4)), dim=1)
        out1_1, out1_2 = self.up1(out)
        out = torch.concat((stage2,out1_2), dim=1)
        out2_1, out2_2 = self.up2(out)
        out = torch.concat((stage1, out2_2), dim=1)
        out3 = self.conv1(out)
        out4 = self.attention(self.conv2(out3))
        out = self.refine(out4, out3, out2_1, out1_1, stage4)
        return out


if __name__ == '__main__':
    input = torch.randn((1, 3, 512, 512))
    net = newnetrrm(3)
    output = net(input)
    print(net)
