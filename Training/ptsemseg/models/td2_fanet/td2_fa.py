import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import resnet18,resnet34,resnet50
from ptsemseg.loss import OhemCELoss2D,SegmentationLosses
from ptsemseg.utils import split_fanet_dict
from ptsemseg.models.td2_psp.pspnet_2p import pspnet_2p
from .transformer import Encoding, Attention
import pdb
import logging


up_kwargs = {'mode': 'bilinear', 'align_corners': True}
logger = logging.getLogger("ptsemseg")

class td2_fa(nn.Module):
    def __init__(self,
                 nclass=21,
                 backbone='resnet18',
                 norm_layer=None,
                 loss_fn=None,
                 path_num=None,
                 mdl_path = None,
                 teacher = None):
        super(td2_fa, self).__init__()

        self.loss_fn = loss_fn
        self.fa_path = mdl_path
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
            self.expansion = 1
            ResNet_ = resnet18
        elif backbone == 'resnet34':
            self.expansion = 1
            ResNet_ = resnet34
        elif backbone == 'resnet50':
            self.expansion = 4
            ResNet_ = resnet50
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))
        # bilinear upsample options


        self.pretrained1 = ResNet_(norm_layer=norm_layer)
        self.pretrained2 = ResNet_(norm_layer=norm_layer)

        self.ffm_32_1 = FAModule(512*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_16_1 = FAModule(256*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_8_1 = FAModule(128*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_4_1 = FAModule(64*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_32_2 = FAModule(512*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_16_2 = FAModule(256*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_8_2 = FAModule(128*self.expansion,256,128,norm_layer=norm_layer)
        self.ffm_4_2 = FAModule(64*self.expansion,256,128,norm_layer=norm_layer)


        self.enc1 = Encoding(256,64,256,norm_layer)
        self.enc2 = Encoding(256,64,256,norm_layer)
        self.atn1 = Attention(256,64,norm_layer)
        self.atn2 = Attention(256,64,norm_layer)

        self.layer_norm1 = Layer_Norm([96, 192])
        self.layer_norm2 = Layer_Norm([96, 192])

        self.head1 = FPNOutput(256,256, nclass, norm_layer)
        self.head2 = FPNOutput(256,256, nclass, norm_layer)

        self.head_aux1 = FPNOutput(128,64, nclass, norm_layer)
        self.head_aux2 = FPNOutput(128,64, nclass, norm_layer)

        pdb.set_trace()
        self.pretrained_init()
        self.KLD = nn.KLDivLoss()
        self.get_params()
        self.teacher = teacher


    def forward_path1(self, f_img):
        f1_img = f_img[0]
        f2_img = f_img[1]
        
        _, _, h, w = f2_img.size()

        # Subnet-1 forward
        feat4_1, feat8_1, feat16_1, feat32_1 = self.pretrained1(f2_img)
        upfeat_32_1 = self.ffm_32_1(feat32_1,None,True,True)
        upfeat_16_1, smfeat_16_1 = self.ffm_16_1(feat16_1,upfeat_32_1,True,True)
        upfeat_8_1 = self.ffm_8_1(feat8_1,upfeat_16_1,True,False)
        smfeat_4_1 = self.ffm_4_1(feat4_1,upfeat_8_1,False,True)
        z1 = self._upsample_cat(smfeat_16_1, smfeat_4_1)


        # Subnet-2 forward
        feat4_2, feat8_2, feat16_2, feat32_2 = self.pretrained2(f1_img)
        upfeat_32_2 = self.ffm_32_2(feat32_2,None,True,True)
        upfeat_16_2, smfeat_16_2 = self.ffm_16_2(feat16_2,upfeat_32_2,True,True)
        upfeat_8_2 = self.ffm_8_2(feat8_2,upfeat_16_2,True,False)
        smfeat_4_2 = self.ffm_4_2(feat4_2,upfeat_8_2,False,True)
        z2 = self._upsample_cat(smfeat_16_2, smfeat_4_2)

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

            return outputs1, outputs1_sub, KD_loss1
        else:
            return outputs1


    def forward_path2(self, f_img):
        f1_img = f_img[0]
        f2_img = f_img[1]
        
        _, _, h, w = f2_img.size()


        # Subnet-1 forward
        feat4_1, feat8_1, feat16_1, feat32_1 = self.pretrained1(f1_img)
        upfeat_32_1 = self.ffm_32_1(feat32_1,None,True,True)
        upfeat_16_1, smfeat_16_1 = self.ffm_16_1(feat16_1,upfeat_32_1,True,True)
        upfeat_8_1 = self.ffm_8_1(feat8_1,upfeat_16_1,True,False)
        smfeat_4_1 = self.ffm_4_1(feat4_1,upfeat_8_1,False,True)
        z1 = self._upsample_cat(smfeat_16_1, smfeat_4_1)

        
        # Subnet-2 forward
        feat4_2, feat8_2, feat16_2, feat32_2 = self.pretrained2(f2_img)
        upfeat_32_2 = self.ffm_32_2(feat32_2,None,True,True)
        upfeat_16_2, smfeat_16_2 = self.ffm_16_2(feat16_2,upfeat_32_2,True,True)
        upfeat_8_2 = self.ffm_8_2(feat8_2,upfeat_16_2,True,False)
        smfeat_4_2 = self.ffm_4_2(feat4_2,upfeat_8_2,False,True)
        z2 = self._upsample_cat(smfeat_16_2, smfeat_4_2)

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


            return outputs2, outputs2_sub, KD_loss2
        else:
            return outputs2




    def _upsample_cat(self, x1, x2):
        '''Upsample and concatenate feature maps.
        '''
        _,_,H,W = x2.size()
        x1 = F.interpolate(x1, (H,W), **self._up_kwargs)
        x = torch.cat([x1,x2],dim=1)
        return x


    def forward(self, f2_img, lbl=None, pos_id=None):
        
        if pos_id == 0:
            outputs = self.forward_path1(f2_img)
        elif pos_id == 1:
            outputs = self.forward_path2(f2_img)
        else:
            raise RuntimeError("Only Two Paths.")

        if self.training:
            outputs_, outputs_sub, KD_loss = outputs
            loss = self.loss_fn(outputs_,lbl) +\
                    0.5*self.loss_fn(outputs_sub,lbl) +\
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

    def pretrained_init(self):

        if self.fa_path is not None:
            if os.path.isfile(self.fa_path):
                logger.info("Initializaing sub networks with pretrained '{}'".format(self.fa_path))
                print("Initializaing sub networks with pretrained '{}'".format(self.fa_path))
                
                model_state = torch.load(self.fa_path)
                backbone_state, ffm_32_state, ffm_16_state, ffm_8_state, ffm_4_state, output_state , output_aux_state = split_fanet_dict(model_state,self.path_num)

                self.pretrained1.load_state_dict(backbone_state, strict=True)
                self.pretrained2.load_state_dict(backbone_state, strict=True)
                self.ffm_32_1.load_state_dict(ffm_32_state, strict=True)
                self.ffm_32_2.load_state_dict(ffm_32_state, strict=True)

                self.ffm_16_1.load_state_dict(ffm_16_state, strict=True)
                self.ffm_16_2.load_state_dict(ffm_16_state, strict=True)

                self.ffm_8_1.load_state_dict(ffm_8_state, strict=True)
                self.ffm_8_2.load_state_dict(ffm_8_state, strict=True)

                self.ffm_4_1.load_state_dict(ffm_4_state, strict=True)
                self.ffm_4_2.load_state_dict(ffm_4_state, strict=True)

                self.head1.load_state_dict(output_state, strict=True)
                self.head2.load_state_dict(output_state, strict=True)

                self.head_aux1.load_state_dict(output_aux_state, strict=True)
                self.head_aux2.load_state_dict(output_aux_state, strict=True)

            else:
                logger.info("No pretrained found at '{}'".format(self.fa_path))

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if isinstance(child, (OhemCELoss2D,SegmentationLosses,nn.KLDivLoss,pspnet_2p)):
                continue
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (Encoding, Attention, FAModule, FPNOutput, Layer_Norm)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, activation='leaky_relu',*args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer = norm_layer
        if self.norm_layer is not None:
            self.bn = norm_layer(out_chan, activation=activation)
        else:
            self.bn =  lambda x:x

        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FPNOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None, *args, **kwargs):
        super(FPNOutput, self).__init__()
        self.norm_layer = norm_layer
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
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
            elif isinstance(module, self.norm_layer):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FAModule(nn.Module):
    def __init__(self, in_chan, mid_chn=256, out_chan=128, norm_layer=None, *args, **kwargs):
        super(FAModule, self).__init__()
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        mid_chn = int(in_chan/2)        
        self.w_qs = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_ks = ConvBNReLU(in_chan, 32, ks=1, stride=1, padding=0, norm_layer=norm_layer, activation='none')

        self.w_vs = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.latlayer3 = ConvBNReLU(in_chan, in_chan, ks=1, stride=1, padding=0, norm_layer=norm_layer)

        self.up = ConvBNReLU(in_chan, mid_chn, ks=1, stride=1, padding=1, norm_layer=norm_layer)
        self.smooth = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=norm_layer)

        self.init_weight()

    def forward(self, feat, up_fea_in,up_flag, smf_flag):

        query = self.w_qs(feat)
        key   = self.w_ks(feat)
        value = self.w_vs(feat)

        N,C,H,W = feat.size()

        query_ = query.view(N,32,-1).permute(0, 2, 1)
        query = F.normalize(query_, p=2, dim=2, eps=1e-12)

        key_   = key.view(N,32,-1)
        key   = F.normalize(key_, p=2, dim=1, eps=1e-12)

        value = value.view(N,C,-1).permute(0, 2, 1)

        f = torch.matmul(key, value)
        y = torch.matmul(query, f)
        #pdb.set_trace()
        y = y.permute(0, 2, 1).contiguous()

        y = y.view(N, C, H, W)
        W_y = self.latlayer3(y)
        p_feat = W_y + feat

        if up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            if up_fea_in is not None:
                smooth_feat = self.smooth(p_feat)
                return up_feat, smooth_feat
            else: 
                return up_feat

        if up_flag and not smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            up_feat = self.up(p_feat)
            return up_feat

        if not up_flag and smf_flag:
            if up_fea_in is not None:
                p_feat = self._upsample_add(up_fea_in, p_feat)
            smooth_feat = self.smooth(p_feat)
            return smooth_feat


    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        '''
        _,_,H,W = y.size()
        return F.interpolate(x, (H,W), **self._up_kwargs) + y

    def init_weight(self):
        for ly in self.children():
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
            elif isinstance(module, self.norm_layer):
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
