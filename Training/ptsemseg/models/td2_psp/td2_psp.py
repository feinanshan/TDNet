import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34,resnet50
import random
from ptsemseg.utils import split_psp_dict
from ptsemseg.models.td2_psp.pspnet_2p import pspnet_2p
import logging
import pdb
import os
from ptsemseg.loss import OhemCELoss2D,SegmentationLosses
from .transformer import Encoding, Attention

up_kwargs = {'mode': 'bilinear', 'align_corners': True}
logger = logging.getLogger("ptsemseg")

class td2_psp(nn.Module):
    """
    """
    def __init__(self,
                 nclass=21,
                 norm_layer=nn.BatchNorm2d,
                 backbone='resnet101',
                 dilated=True,
                 aux=True,
                 multi_grid=True,
                 loss_fn=None,
                 path_num=None,
                 mdl_path = None,
                 teacher = None
                 ):
        super(td2_psp, self).__init__()

        self.psp_path = mdl_path
        self.loss_fn = loss_fn
        self.path_num = path_num
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass

        # copying modules from pretrained models
        self.backbone = backbone
        assert(backbone == 'resnet50' or backbone == 'resnet34' or backbone == 'resnet18')
        assert(path_num == 2)

        if backbone == 'resnet18':
            ResNet_ = resnet18
            deep_base = False
            self.expansion = 1
        elif backbone == 'resnet34':
            ResNet_ = resnet34
            deep_base = False
            self.expansion = 1
        elif backbone == 'resnet50':
            ResNet_ = resnet50
            deep_base = True
            self.expansion = 4
        else:
            raise RuntimeError("Four branch model only support ResNet18 amd ResNet34")

        self.pretrained1 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        self.pretrained2 = ResNet_(pretrained=True, dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        # bilinear upsample options

        self.psp1 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num, pid=0)
        self.psp2 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num, pid=1)

        self.enc1 = Encoding(512*self.expansion,64,512*self.expansion//4,norm_layer)
        self.enc2 = Encoding(512*self.expansion,64,512*self.expansion//4,norm_layer)
        self.atn1 = Attention(512*self.expansion//4,64,norm_layer)
        self.atn2 = Attention(512*self.expansion//4,64,norm_layer)

        self.layer_norm1 = Layer_Norm([97, 193])
        self.layer_norm2 = Layer_Norm([97, 193])

        self.head1 = FCNHead(512*self.expansion//4, nclass, norm_layer, chn_down=2)
        self.head2 = FCNHead(512*self.expansion//4, nclass, norm_layer, chn_down=2)


        if aux:
            self.auxlayer1 = FCNHead(256*self.expansion, nclass, norm_layer)
            self.auxlayer2 = FCNHead(256*self.expansion, nclass, norm_layer)

        self.pretrained_init()
        self.KLD = nn.KLDivLoss()
        self.get_params()
        self.teacher = teacher
     
    
    def forward_path1(self, f_img):
        f1_img = f_img[0]
        f2_img = f_img[1]
        
        _, _, h, w = f2_img.size()

        c3_1,c4_1 = self.pretrained1(f2_img)
        c3_2,c4_2 = self.pretrained2(f1_img)

        z1 = self.psp1(c4_1)
        z2 = self.psp2(c4_2)

        q1, v1 = self.enc1(z1, pre=False)
        k2_, v2_ = self.enc2(z2, pre=True)

        atn_1 = self.atn1(k2_, v2_, q1, fea_size=z1.size())
        out1 = self.head1(self.layer_norm1(atn_1 + v1))
        out1_sub = self.head1(self.layer_norm1(v1))

        outputs1 = F.interpolate(out1, (h, w), **self._up_kwargs)
        outputs1_sub = F.interpolate(out1_sub, (h, w), **self._up_kwargs)
        
        if self.training:
            #############Knowledge-distillation###########
            self.teacher.eval()
            T_logit_12, T_logit_1, T_logit_2 = self.teacher(f2_img)
            T_logit_12 = T_logit_12.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()
           
            KD_loss1 = self.KLDive_loss(out1,T_logit_12)+ 0.5*self.KLDive_loss(out1_sub,T_logit_1)
            auxout1 = self.auxlayer1(c3_1)        
            auxout1 = F.interpolate(auxout1, (h, w), **self._up_kwargs)
            return outputs1, outputs1_sub, auxout1, KD_loss1
        else:
            return outputs1
        
    def forward_path2(self, f_img):
        f1_img = f_img[0]
        f2_img = f_img[1]
        
        _, _, h, w = f2_img.size()

        c3_1,c4_1 = self.pretrained1(f1_img)
        c3_2,c4_2 = self.pretrained2(f2_img)

        z1 = self.psp1(c4_1)
        z2 = self.psp2(c4_2)

        k1_, v1_ = self.enc1(z1, pre=True)
        q2,  v2 = self.enc2(z2, pre=False)

        atn_2 = self.atn2(k1_, v1_, q2, fea_size=z2.size())
        out2 = self.head2(self.layer_norm2(atn_2 + v2))
        out2_sub = self.head2(self.layer_norm2(v2))

        outputs2 = F.interpolate(out2, (h, w), **self._up_kwargs)
        outputs2_sub = F.interpolate(out2_sub, (h, w), **self._up_kwargs)
        
        if self.training:
            #############Knowledge-distillation###########
            self.teacher.eval()
            T_logit_12, T_logit_1, T_logit_2 = self.teacher(f2_img)
            T_logit_12 = T_logit_12.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()

            KD_loss2 = self.KLDive_loss(out2,T_logit_12)+ 0.5*self.KLDive_loss(out2_sub,T_logit_2)

            auxout2 = self.auxlayer2(c3_2)
            auxout2 = F.interpolate(auxout2, (h, w), **self._up_kwargs)
            return outputs2, outputs2_sub, auxout2, KD_loss2
        else:
            return outputs2

        
    def forward(self, f2_img, lbl=None, pos_id=None):
        
        if pos_id == 0:
            outputs = self.forward_path1(f2_img)
        elif pos_id == 1:
            outputs = self.forward_path2(f2_img)
        else:
            raise RuntimeError("Only Two Paths.")

        if self.training:
            outputs_, outputs_sub, auxout, KD_loss = outputs
            loss = self.loss_fn(outputs_,lbl) +\
                    0.5*self.loss_fn(outputs_sub,lbl) +\
                    0.1*self.loss_fn(auxout,lbl) +\
                    1*KD_loss

            return loss
        else:
            return outputs

        
    def KLDive_loss(self, Q, P):
        # Info_gain - KL Divergence
        # P is the target
        # Q is the computed one
        temp = 1
        P = F.softmax(P/temp,dim=1)+1e-8 
        Q = F.softmax(Q/temp,dim=1)+1e-8 
    
        KLDiv = (P * (P/Q).log()).sum(1)* (temp**2)
        return KLDiv.mean()

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child, (OhemCELoss2D, SegmentationLosses, pspnet_2p, nn.KLDivLoss)):
                continue
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (Encoding, Attention, PyramidPooling, FCNHead, Layer_Norm)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


    def pretrained_init(self):

        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                logger.info("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
                print("Initializaing sub networks with pretrained '{}'".format(self.psp_path))
                
                model_state = torch.load(self.psp_path)
                backbone_state, psp_state, head_state1, head_state2, _, _, auxlayer_state = split_psp_dict(model_state,self.path_num)

                self.pretrained1.load_state_dict(backbone_state, strict=True)
                self.pretrained2.load_state_dict(backbone_state, strict=True)
                self.psp1.load_state_dict(psp_state, strict=True)
                self.psp2.load_state_dict(psp_state, strict=True)
                self.auxlayer1.load_state_dict(auxlayer_state, strict=True)
                self.auxlayer2.load_state_dict(auxlayer_state, strict=True)

            else:
                logger.info("No pretrained found at '{}'".format(self.psp_path))

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
        self.init_weight()

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

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs={}, chn_down=4):
        super(FCNHead, self).__init__()

        inter_channels = in_channels // chn_down

        self._up_kwargs = up_kwargs
        self.norm_layer = norm_layer
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))
        self.init_weight()

    def forward(self, x):
        return self.conv5(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Sequential):
                for lz in ly:
                    if isinstance(lz, nn.Conv2d):
                        nn.init.kaiming_normal_(lz.weight, a=1)
                        if not lz.bias is None: nn.init.constant_(lz.bias, 0)
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

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

class Layer_Norm(nn.Module):
    def __init__(self, shape):
        super(Layer_Norm, self).__init__()
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        return self.ln(x)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, (nn.LayerNorm)):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
