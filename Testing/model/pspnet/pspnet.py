import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34,resnet50,resnet101
import os

from torch.autograd import Variable

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class BatchNorm2d(nn.BatchNorm2d):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, num_features, activation='none'):
        super(BatchNorm2d, self).__init__(num_features=num_features)
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'none':
            self.activation = lambda x:x
        else:
            raise Exception("Accepted activation: ['leaky_relu']")

    def forward(self, x):
        return self.activation(super(BatchNorm2d, self).forward(x))


up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class pspnet(nn.Module):

    def __init__(self,
            nclass=21,
            norm_layer=BatchNorm2d,
            backbone='resnet101',
            dilated=True,
            aux=True,
            multi_grid=True,
            model_path=None,
        ):
        super(pspnet, self).__init__()
        self.psp_path = model_path
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        # copying modules from pretrained models
        self.backbone = backbone
        
        if backbone == 'resnet18':
            self.pretrained = resnet18(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.pretrained = resnet34(dilated=dilated, multi_grid=multi_grid,
                                               deep_base=False)
            self.expansion = 1
        elif backbone == 'resnet50':
            self.pretrained = resnet50(dilated=dilated,multi_grid=multi_grid,
                                              norm_layer=norm_layer)
            self.expansion = 4
        elif backbone == 'resnet101':
            self.pretrained = resnet101(dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer)
            self.expansion = 4
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        
        self.head = PSPHead(512*self.expansion, nclass, norm_layer, self._up_kwargs)
        self.pretrained_mp_load()

    def forward(self, x, pos_id=None):
        x = x[-1:]
        _, _, h, w = x.size()

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)

        x = self.head(c4)
        outputs = F.interpolate(x, (h,w), **self._up_kwargs)

        return outputs

    def pretrained_mp_load(self):
        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                print("Loading pretrained model from '{}'".format(self.psp_path))
                model_state = torch.load(self.psp_path)
                self.load_state_dict(model_state, strict=True)

            else:
                print("No pretrained found at '{}'".format(self.psp_path))


class PSPHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSPHead, self).__init__()
        inter_channels = in_channels // 4

        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(in_channels * 2, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
    def forward(self, x):
        return self.conv5(x)


class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)



class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, with_global=False):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)
