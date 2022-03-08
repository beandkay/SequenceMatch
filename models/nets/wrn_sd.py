import math
import torch
import torch.nn as nn
import torch.nn.functional as F

momentum = 0.001


def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))


class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

def ScalaNet(channel_in, channel_out, size):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=size, stride=size),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        nn.AvgPool2d(4, 4)
        )

class SepConv(nn.Module):

    def __init__(self, channel_in, channel_out, kernel_size=3, stride=2, padding=1, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_in, channel_in, kernel_size=kernel_size, stride=1, padding=padding, groups=channel_in, bias=False),
            nn.Conv2d(channel_in, channel_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_out, affine=affine),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.op(x)

def dowmsampleBottleneck(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU(),
        )

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetSD(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False):
        super(WideResNetSD, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        
        self.scala1 = nn.Sequential(
            SepConv(
                channel_in=channels[1],
                channel_out=channels[2]
            ),
            SepConv(
                channel_in=channels[2],
                channel_out=channels[3]
            ),
            nn.AvgPool2d(4, 4)
        )

        self.scala2 = nn.Sequential(
            SepConv(
                channel_in=channels[2],
                channel_out=channels[3],
            ),
            nn.AvgPool2d(4, 4)
        )
        
        self.scala3 = nn.AvgPool2d(4, 4)
        
        self.attention1 = nn.Sequential(
            SepConv(
                channel_in=channels[1],
                channel_out=channels[1]
            ),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        self.attention2 = nn.Sequential(
            SepConv(
                channel_in=channels[2],
                channel_out=channels[2]
            ),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        )

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.bn2 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.bn3 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc1 = nn.Linear(channels[3], num_classes)
        self.fc2 = nn.Linear(channels[3], num_classes)
        self.fc3 = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier1 = nn.Linear(self.channels, 4)
            self.rot_classifier2 = nn.Linear(self.channels, 4)
            self.rot_classifier3 = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x, ood_test=False):
        feature_list = []
        out = self.conv1(x)
        out = self.block1(out)

        fea1 = self.attention1(out)
        fea1 = fea1 * out
        feature_list.append(fea1)

        out = self.block2(out)

        fea2 = self.attention2(out)
        fea2 = fea2 * out
        feature_list.append(fea2)

        out = self.block3(out)
        feature_list.append(out)

        out1_feature = self.scala1(feature_list[0])
        out1_feature = self.relu(self.bn1(out1_feature))
        out1_feature = F.adaptive_avg_pool2d(out1_feature, 1)
        out1_feature = out1_feature.view(-1, self.channels)
        
        out2_feature = self.scala2(feature_list[1])
        out2_feature = self.relu(self.bn2(out2_feature))
        out2_feature = F.adaptive_avg_pool2d(out2_feature, 1)
        out2_feature = out2_feature.view(-1, self.channels)

        out3_feature = self.scala3(feature_list[2])
        out3_feature = self.relu(self.bn3(out3_feature))
        out3_feature = F.adaptive_avg_pool2d(out3_feature, 1)
        out3_feature = out3_feature.view(-1, self.channels)
        
        output1 = self.fc1(out1_feature)
        output2 = self.fc2(out2_feature)
        output3 = self.fc3(out3_feature)
        
        teacher_feature = out3_feature.detach()
        feature_loss = ((teacher_feature - out2_feature)**2 + (teacher_feature - out1_feature)**2).sum()
        
        if ood_test:
            return [output1, output2, output3], [out1_feature, out2_feature, out3_feature]
        else:
            if self.is_remix:
                rot_output1 = self.rot_classifier1(out1_feature)
                rot_output2 = self.rot_classifier2(out2_feature)
                rot_output3 = self.rot_classifier3(out3_feature)
                return [output1, output2, output3], [rot_output1, rot_output2, rot_output3], feature_loss
            else:
                return [output1, output2, output3], feature_loss

class build_WideResNetSD:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return WideResNetSD(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )


if __name__ == '__main__':
    wrn_builder = build_WideResNetSD(1, 10, 2, 0.01, 0.1, 0.5)
    wrn = wrn_builder.build(10)
    print(wrn)