import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from .resnet import resnet50,resnet101,resnet152
import pdb
import logging
from ptsemseg.utils import split_psp_state_dict,convert_state_dict

logger = logging.getLogger("ptsemseg")

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class pspnet_4p(nn.Module):

    """
    Pyramid Scene Parsing Network
    URL: https://arxiv.org/abs/1612.01105

    References:
    1) Original Author's code: https://github.com/hszhao/PSPNet
    2) Chainer implementation by @mitmul: https://github.com/mitmul/chainer-pspnet
    3) TensorFlow implementation by @hellochick: https://github.com/hellochick/PSPNet-tensorflow

    Visualization:
    http://dgschwend.github.io/netscope/#/gist/6bfb59e6a3cfcb4e2bb8d47f827c2928

    """

    def __init__(self,
            nclass=21,
            se_loss=None,
            norm_layer=nn.BatchNorm2d,
            backbone='resnet101',
            root='~/.encoding/models',
            dilated=True,
            aux=True,
            multi_grid=True,
            fixed=True,
            loss_fn=None,
            teacher_model=None,
            path_num=None
        ):
        super(pspnet_4p, self).__init__()

        self.path_num = path_num
        self.fixed = fixed
        self.teacher_model = teacher_model
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        # copying modules from pretrained models
        self.backbone = backbone
        if backbone == 'resnet50':
            self.pretrained = resnet50(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                              norm_layer=norm_layer, root=root)
        elif backbone == 'resnet101':
            self.pretrained = resnet101(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer, root=root)
        elif backbone == 'resnet152':
            self.pretrained = resnet152(pretrained=True, dilated=dilated,multi_grid=multi_grid,
                                               norm_layer=norm_layer, root=root)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options
        
        #self.head = PSPHead(2048, nclass, norm_layer, self._up_kwargs)


        self.psp1 =  PyramidPooling(2048, norm_layer, self._up_kwargs, path_num=self.path_num, pid=0)
        self.psp2 =  PyramidPooling(2048, norm_layer, self._up_kwargs, path_num=self.path_num, pid=1)
        self.psp3 =  PyramidPooling(2048, norm_layer, self._up_kwargs, path_num=self.path_num, pid=2)
        self.psp4 =  PyramidPooling(2048, norm_layer, self._up_kwargs, path_num=self.path_num, pid=3)
    
        self.group1 = ConvBNReLU(1024, 512, norm_layer, ksize=3, pad=1, BNLU=False)
        self.group2 = ConvBNReLU(1024, 512, norm_layer, ksize=3, pad=1, BNLU=False)
        self.group3 = ConvBNReLU(1024, 512, norm_layer, ksize=3, pad=1, BNLU=False)
        self.group4 = ConvBNReLU(1024, 512, norm_layer, ksize=3, pad=1, BNLU=False)

        self.head = PredLayer(512, nclass, norm_layer)

        self.pretrained_init()

    def forward(self, x,test_mod=False):
        _, _, h, w = x.size()

        c4 = self.pretrained(x,aux=False)
        p1 = self.psp1(c4)
        p2 = self.psp2(c4)
        p3 = self.psp3(c4)
        p4 = self.psp4(c4)
        g1 = self.group1(p1)
        g2 = self.group2(p2)
        g3 = self.group3(p3)
        g4 = self.group4(p4)
        out12 = self.head(g1+g2+g3+g4)
        out1 = self.head(g1+g1+g1+g1)
        out2 = self.head(g2+g2+g2+g2)
        out3 = self.head(g3+g3+g3+g3)
        out4 = self.head(g4+g4+g4+g4)
        return out12, out1, out3, out2, out4

    def pretrained_init(self):

        if self.teacher_model is not None:
            if os.path.isfile(self.teacher_model):
                logger.info("Initializing Teacher with pretrained '{}'".format(self.teacher_model))
                print("Initializing Teacher with pretrained '{}'".format(self.teacher_model))
                model_state = torch.load(self.teacher_model)
                backbone_state, psp_state, grp_state1, grp_state2, grp_state3, grp_state4, head_state, auxlayer_state = split_psp_state_dict(model_state,self.path_num)
                self.pretrained.load_state_dict(backbone_state, strict=True)
                self.psp1.load_state_dict(psp_state, strict=True)
                self.psp2.load_state_dict(psp_state, strict=True)
                self.psp3.load_state_dict(psp_state, strict=True)
                self.psp4.load_state_dict(psp_state, strict=True)
                self.group1.load_state_dict(grp_state1, strict=True)
                self.group2.load_state_dict(grp_state2, strict=True)
                self.group3.load_state_dict(grp_state3, strict=True)
                self.group4.load_state_dict(grp_state4, strict=True)
                self.head.load_state_dict(head_state, strict=True)
            else:
                logger.info("No pretrained found at '{}'".format(self.teacher_model))
        
            if self.fixed:
                for param in self.parameters():
                    param.requires_grad = False






class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """

    def __init__(self, in_channels, norm_layer, up_kwargs, path_num=None, pid=None):
        super(PyramidPooling, self).__init__()
        self.norm_layer = norm_layer
        self.pid = pid
        self.path_num = path_num
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels / 4)
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
        n, c, h, w = x.size()
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)

        x = x[:, self.pid*c//self.path_num:(self.pid+1)*c//self.path_num]
        feat1 = feat1[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat2 = feat2[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat3 = feat3[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]
        feat4 = feat4[:, self.pid*c//(self.path_num*4):(self.pid+1)*c//(self.path_num*4)]

        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, ksize=3, pad=1, BNLU=False):
        super(ConvBNReLU, self).__init__()
        self.norm_layer = norm_layer
        if BNLU:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU())
        else:
            self.conv5 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad, bias=False))

    def forward(self, x):
        return self.conv5(x)


class PredLayer(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(PredLayer, self).__init__()
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(norm_layer(in_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)


