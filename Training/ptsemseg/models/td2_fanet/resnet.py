#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18','resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_chan, out_chan, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = norm_layer(out_chan, activation='leaky_relu')
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = norm_layer(out_chan, activation='none')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_chan, activation='none'),
                )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut + out
        out_ = self.relu(out_)
        return out_

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_chan, out_chan, stride=1, base_width=64, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = int(out_chan*(base_width / 64.)) * 1
        self.norm_layer = norm_layer
        self.conv1 = conv1x1(in_chan, width)
        self.bn1 = norm_layer(width, activation='leaky_relu')
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = norm_layer(width, activation='leaky_relu')
        self.conv3 = conv1x1(width, out_chan * self.expansion)
        self.bn3 = norm_layer(out_chan * self.expansion, activation='none')
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan*self.expansion or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan*self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(out_chan*self.expansion, activation='none'),
                )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.conv3(out)
        out = self.bn3(out)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out_ = shortcut +out
        out_ = self.relu(out_)

        return out_

class ResNet(nn.Module):
    def __init__(self, block, layers, strides, norm_layer=None):
        super(ResNet, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64, activation='leaky_relu')
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inplanes = 64
        self.layer1 = self.create_layer(block,   64, bnum=layers[0], stride=strides[0], norm_layer=norm_layer)
        self.layer2 = self.create_layer(block,  128, bnum=layers[1], stride=strides[1], norm_layer=norm_layer)
        self.layer3 = self.create_layer(block,  256, bnum=layers[2], stride=strides[2], norm_layer=norm_layer)
        self.layer4 = self.create_layer(block,  512, bnum=layers[3], stride=strides[3], norm_layer=norm_layer)

    def create_layer(self, block , out_chan, bnum, stride=1,norm_layer=None):
        layers = [block(self.inplanes, out_chan, stride=stride, norm_layer=norm_layer)]
        self.inplanes = out_chan*block.expansion
        for i in range(bnum-1):
            layers.append(block(self.inplanes, out_chan, stride=1, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat4, feat8, feat16, feat32


    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (self.norm_layer)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params

    def init_weight(self,state_dict):
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=True)

def resnet18(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2],[2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3],[2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3],[2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3],[2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=True, norm_layer=None, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3],[2, 2, 2, 2], norm_layer=norm_layer)
    if pretrained:
        model.init_weight(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == "__main__":
    net = Resnet18()
    x = torch.randn(16, 3, 224, 224)
    out = net(x)
    print(out[0].size())
    print(out[1].size())
    print(out[2].size())
    net.get_params()
