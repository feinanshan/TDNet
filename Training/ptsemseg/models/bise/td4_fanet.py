#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import Resnet18,Resnet34,Resnet50,Resnet101,Resnet152
from ptsemseg.loss import OhemCELoss2D,SegmentationLosses
from ptsemseg.utils import split_1p_state_dict, split_2p_state_dict
from ptsemseg.models.bisenet import BiSeNet
from ptsemseg.models.pspnet import pspnet
from .transformer import Encoding, Attention
import logging

logger = logging.getLogger("ptsemseg")
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BiSeNet_4p(nn.Module):
    def __init__(self,
                 nclass,
                 backbone='resnet18',
                 norm_layer=None,
                 loss_fn=None,
                 path_num=None,
                 bise_path = None,
                 teacher = None
                 ):

        super(BiSeNet_4p, self).__init__()

        self._up_kwargs = up_kwargs
        self.bise_path = bise_path
        self.path_num = path_num
        self.loss_fn = loss_fn
        self.norm_layer = norm_layer

        self.cp1 = ContextPath(backbone,norm_layer=norm_layer)
        self.cp2 = ContextPath(backbone,norm_layer=norm_layer)
        self.cp3 = ContextPath(backbone,norm_layer=norm_layer)
        self.cp4 = ContextPath(backbone,norm_layer=norm_layer)

        self.ffm1 = FeatureFusionModule(256, 256,norm_layer=norm_layer, path_num=self.path_num//2, pid=0)
        self.ffm2 = FeatureFusionModule(256, 256,norm_layer=norm_layer, path_num=self.path_num//2, pid=1)
        self.ffm3 = FeatureFusionModule(256, 256,norm_layer=norm_layer, path_num=self.path_num//2, pid=0)
        self.ffm4 = FeatureFusionModule(256, 256,norm_layer=norm_layer, path_num=self.path_num//2, pid=1)

        self.enc1 = Encoding(256//2,64,256//2,norm_layer)
        self.enc2 = Encoding(256//2,64,256//2,norm_layer)
        self.enc3 = Encoding(256//2,64,256//2,norm_layer)
        self.enc4 = Encoding(256//2,64,256//2,norm_layer)

        self.atn1_2 = Attention(256//2,64,norm_layer)
        self.atn1_3 = Attention(256//2,64,norm_layer)
        self.atn1_4 = Attention(256//2,64,norm_layer)

        self.atn2_1 = Attention(256//2,64,norm_layer)
        self.atn2_3 = Attention(256//2,64,norm_layer)
        self.atn2_4 = Attention(256//2,64,norm_layer)

        self.atn3_1 = Attention(256//2,64,norm_layer)
        self.atn3_2 = Attention(256//2,64,norm_layer)
        self.atn3_4 = Attention(256//2,64,norm_layer)

        self.atn4_1 = Attention(256//2,64,norm_layer)
        self.atn4_2 = Attention(256//2,64,norm_layer)
        self.atn4_3 = Attention(256//2,64,norm_layer)

        self.layer_norm1_ = Layer_Norm(128,128)
        self.layer_norm2_ = Layer_Norm(128,128)
        self.layer_norm3_ = Layer_Norm(128,128)
        self.layer_norm4_ = Layer_Norm(128,128)

        self.conv_out_1 = BiSeNetOutput(256//2, 256, nclass,norm_layer=norm_layer)
        self.conv_out_2 = BiSeNetOutput(256//2, 256, nclass,norm_layer=norm_layer)
        self.conv_out_3 = BiSeNetOutput(256//2, 256, nclass,norm_layer=norm_layer)
        self.conv_out_4 = BiSeNetOutput(256//2, 256, nclass,norm_layer=norm_layer)

        self.conv_out16_1 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out16_2 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out16_3 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out16_4 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        
        self.conv_out32_1 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out32_2 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out32_3 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)
        self.conv_out32_4 = BiSeNetOutput(128, 64, nclass,norm_layer=norm_layer)

        self.pretrained_mp_load()
        self.KLD = nn.KLDivLoss()
        self.get_params()
        self.teacher = teacher

    def forward_path1(self,f_img):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        f1_img = f_img[0]
        f2_img = f_img[1]
        f3_img = f_img[2]
        f4_img = f_img[3]
        _, _, h, w = f4_img.size()

        feat_res8_1, feat_cp8_1, feat_cp16_1 = self.cp1(f4_img)
        feat_res8_2, feat_cp8_2, feat_cp16_2 = self.cp2(f1_img)
        feat_res8_3, feat_cp8_3, feat_cp16_3 = self.cp3(f2_img)
        feat_res8_4, feat_cp8_4, feat_cp16_4 = self.cp4(f3_img)

        z1 = self.ffm1(feat_res8_1, feat_cp8_1)
        z2 = self.ffm2(feat_res8_2, feat_cp8_2)
        z3 = self.ffm3(feat_res8_3, feat_cp8_3)
        z4 = self.ffm4(feat_res8_4, feat_cp8_4)

        v1, q1 = self.enc1(z1, pre=False)
        k_2, v_2, _ = self.enc2(z2, pre=True, start=True)
        k_3, v_3, q_3 = self.enc3(z3, pre=True)
        k_4, v_4, q_4 = self.enc4(z4, pre=True)

        v_3_ = self.atn1_2(k_2, v_2, q_3, fea_size=None)
        v_4_ = self.atn1_3(k_3, v_3_+v_3, q_4, fea_size=None)
        atn_1 = self.atn1_4(k_4, v_4_+v_4, q1, fea_size=z4.size())

        out1 = self.conv_out_1(self.layer_norm1_(atn_1 + v1))
        outputs1 = F.interpolate(out1, (h, w), **self._up_kwargs)

        if self.training:

            out1_sub = self.conv_out_1(self.layer_norm1_(v1))
            outputs1_sub = F.interpolate(out1_sub, (h, w), **self._up_kwargs)
            #############Knowledge-distillation###########
            self.teacher.eval()
            f4_img = F.interpolate(f4_img, (h+1, w+1), mode='bilinear', align_corners=True)
            T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            T_logit_1234 = T_logit_1234.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()
            T_logit_3 = T_logit_3.detach()
            T_logit_4 = T_logit_4.detach()
            T_logit_1234 = F.interpolate(T_logit_1234, (h, w), mode='bilinear', align_corners=True)
            T_logit_1 = F.interpolate(T_logit_1, (h, w), mode='bilinear', align_corners=True)
            KD_loss1 = self.KLDive_loss(outputs1,T_logit_1234)+ self.KLDive_loss(outputs1_sub,T_logit_1)

            feat_out16_1 = self.conv_out16_1(feat_cp8_1)
            feat_out32_1 = self.conv_out32_1(feat_cp16_1)
            feat_out16_1 = F.interpolate(feat_out16_1, (h, w), mode='bilinear', align_corners=True)
            feat_out32_1 = F.interpolate(feat_out32_1, (h, w), mode='bilinear', align_corners=True)

            return outputs1, outputs1_sub, feat_out16_1, feat_out32_1, KD_loss1
        else:
            return outputs1

    def forward_path2(self,f_img):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        f1_img = f_img[0]
        f2_img = f_img[1]
        f3_img = f_img[2]
        f4_img = f_img[3]
        _, _, h, w = f4_img.size()

        feat_res8_1, feat_cp8_1, feat_cp16_1 = self.cp1(f3_img)
        feat_res8_2, feat_cp8_2, feat_cp16_2 = self.cp2(f4_img)
        feat_res8_3, feat_cp8_3, feat_cp16_3 = self.cp3(f1_img)
        feat_res8_4, feat_cp8_4, feat_cp16_4 = self.cp4(f2_img)

        z1 = self.ffm1(feat_res8_1, feat_cp8_1)
        z2 = self.ffm2(feat_res8_2, feat_cp8_2)
        z3 = self.ffm3(feat_res8_3, feat_cp8_3)
        z4 = self.ffm4(feat_res8_4, feat_cp8_4)

        k_1, v_1, q_1 = self.enc1(z1, pre=True)
        v2, q2 = self.enc2(z2, pre=False)
        k_3, v_3, _ = self.enc3(z3, pre=True, start=True)
        k_4, v_4, q_4 = self.enc4(z4, pre=True)

        v_4_ = self.atn2_3(k_3, v_3, q_4, fea_size=None)
        v_1_ = self.atn2_4(k_4, v_4_+v_4, q_1, fea_size=None)
        atn_2 = self.atn2_1(k_1, v_1_+v_1, q2, fea_size=z4.size())

        out2 = self.conv_out_2(self.layer_norm2_(atn_2 + v2))
        outputs2 = F.interpolate(out2, (h, w), **self._up_kwargs)

        if self.training:

            out2_sub = self.conv_out_2(self.layer_norm2_(v2))
            outputs2_sub = F.interpolate(out2_sub, (h, w), **self._up_kwargs)
            #############Knowledge-distillation###########
            self.teacher.eval()
            f4_img = F.interpolate(f4_img, (h+1, w+1), mode='bilinear', align_corners=True)
            T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            T_logit_1234 = T_logit_1234.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()
            T_logit_3 = T_logit_3.detach()
            T_logit_4 = T_logit_4.detach()
            T_logit_1234 = F.interpolate(T_logit_1234, (h, w), mode='bilinear', align_corners=True)
            T_logit_2 = F.interpolate(T_logit_2, (h, w), mode='bilinear', align_corners=True)
            KD_loss2 = self.KLDive_loss(outputs2,T_logit_1234)+ self.KLDive_loss(outputs2_sub,T_logit_2)

            feat_out16_2 = self.conv_out16_2(feat_cp8_2)
            feat_out32_2 = self.conv_out32_2(feat_cp16_2)
            feat_out16_2 = F.interpolate(feat_out16_2, (h, w), mode='bilinear', align_corners=True)
            feat_out32_2 = F.interpolate(feat_out32_2, (h, w), mode='bilinear', align_corners=True)

            return outputs2, outputs2_sub, feat_out16_2, feat_out32_2, KD_loss2
        else:
            return outputs2

    def forward_path3(self, f_img):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        f1_img = f_img[0]
        f2_img = f_img[1]
        f3_img = f_img[2]
        f4_img = f_img[3]
        _, _, h, w = f4_img.size()

        feat_res8_1, feat_cp8_1, feat_cp16_1 = self.cp1(f2_img)
        feat_res8_2, feat_cp8_2, feat_cp16_2 = self.cp2(f3_img)
        feat_res8_3, feat_cp8_3, feat_cp16_3 = self.cp3(f4_img)
        feat_res8_4, feat_cp8_4, feat_cp16_4 = self.cp4(f1_img)

        z1 = self.ffm1(feat_res8_1, feat_cp8_1)
        z2 = self.ffm2(feat_res8_2, feat_cp8_2)
        z3 = self.ffm3(feat_res8_3, feat_cp8_3)
        z4 = self.ffm4(feat_res8_4, feat_cp8_4)

        k_1, v_1, q_1 = self.enc1(z1, pre=True)
        k_2, v_2, q_2 = self.enc2(z2, pre=True)
        v3, q3 = self.enc3(z3, pre=False)
        k_4, v_4, _ = self.enc4(z4, pre=True, start=True)

        v_1_ = self.atn3_4(k_4, v_4, q_1, fea_size=None)
        v_2_ = self.atn3_1(k_1, v_1_+v_1, q_2, fea_size=None)
        atn_3 = self.atn3_2(k_2, v_2_+v_2, q3, fea_size=z4.size())

        out3 = self.conv_out_3(self.layer_norm3_(atn_3 + v3))
        outputs3 = F.interpolate(out3, (h, w), **self._up_kwargs)

        if self.training:

            out3_sub = self.conv_out_3(self.layer_norm3_(v3))
            outputs3_sub = F.interpolate(out3_sub, (h, w), **self._up_kwargs)
            #############Knowledge-distillation###########
            self.teacher.eval()
            f4_img = F.interpolate(f4_img, (h+1, w+1), mode='bilinear', align_corners=True)
            T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            T_logit_1234 = T_logit_1234.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()
            T_logit_3 = T_logit_3.detach()
            T_logit_4 = T_logit_4.detach()
            T_logit_1234 = F.interpolate(T_logit_1234, (h, w), mode='bilinear', align_corners=True)
            T_logit_3 = F.interpolate(T_logit_3, (h, w), mode='bilinear', align_corners=True)
            KD_loss3 = self.KLDive_loss(outputs3,T_logit_1234)+ self.KLDive_loss(outputs3_sub,T_logit_3)

            feat_out16_3 = self.conv_out16_3(feat_cp8_3)
            feat_out32_3 = self.conv_out32_3(feat_cp16_3)
            feat_out16_3 = F.interpolate(feat_out16_3, (h, w), mode='bilinear', align_corners=True)
            feat_out32_3 = F.interpolate(feat_out32_3, (h, w), mode='bilinear', align_corners=True)

            return outputs3, outputs3_sub, feat_out16_3, feat_out32_3, KD_loss3
        else:
            return outputs3

    def forward_path4(self, f_img):
        '''
        :param f_img: [t-3, t-2, t-1, t]
        '''
        f1_img = f_img[0]
        f2_img = f_img[1]
        f3_img = f_img[2]
        f4_img = f_img[3]
        _, _, h, w = f4_img.size()

        feat_res8_1, feat_cp8_1, feat_cp16_1 = self.cp1(f1_img)
        feat_res8_2, feat_cp8_2, feat_cp16_2 = self.cp2(f2_img)
        feat_res8_3, feat_cp8_3, feat_cp16_3 = self.cp3(f3_img)
        feat_res8_4, feat_cp8_4, feat_cp16_4 = self.cp4(f4_img)

        z1 = self.ffm1(feat_res8_1, feat_cp8_1)
        z2 = self.ffm2(feat_res8_2, feat_cp8_2)
        z3 = self.ffm3(feat_res8_3, feat_cp8_3)
        z4 = self.ffm4(feat_res8_4, feat_cp8_4)

        k_1, v_1, _ = self.enc1(z1, pre=True, start=True)
        k_2, v_2, q_2 = self.enc2(z2, pre=True)
        k_3, v_3, q_3 = self.enc3(z3, pre=True)
        v4, q4 = self.enc4(z4, pre=False)

        v_2_ = self.atn4_1(k_1, v_1, q_2, fea_size=None)
        v_3_ = self.atn4_2(k_2, v_2_+v_2, q_3, fea_size=None)
        atn_4 = self.atn4_3(k_3, v_3_+v_3, q4, fea_size=z4.size())

        out4 = self.conv_out_4(self.layer_norm4_(atn_4 + v4))
        outputs4 = F.interpolate(out4, (h, w), **self._up_kwargs)

        if self.training:

            out4_sub = self.conv_out_4(self.layer_norm4_(v4))
            outputs4_sub = F.interpolate(out4_sub, (h, w), **self._up_kwargs)
            #############Knowledge-distillation###########
            self.teacher.eval()
            f4_img = F.interpolate(f4_img, (h+1, w+1), mode='bilinear', align_corners=True)
            T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f4_img)
            T_logit_1234 = T_logit_1234.detach()
            T_logit_1 = T_logit_1.detach()
            T_logit_2 = T_logit_2.detach()
            T_logit_3 = T_logit_3.detach()
            T_logit_4 = T_logit_4.detach()
            T_logit_1234 = F.interpolate(T_logit_1234, (h, w), mode='bilinear', align_corners=True)
            T_logit_4 = F.interpolate(T_logit_4, (h, w), mode='bilinear', align_corners=True)
            KD_loss4 = self.KLDive_loss(outputs4,T_logit_1234)+ self.KLDive_loss(outputs4_sub,T_logit_4)

            feat_out16_4 = self.conv_out16_4(feat_cp8_4)
            feat_out32_4 = self.conv_out32_4(feat_cp16_4)
            feat_out16_4 = F.interpolate(feat_out16_4, (h, w), mode='bilinear', align_corners=True)
            feat_out32_4 = F.interpolate(feat_out32_4, (h, w), mode='bilinear', align_corners=True)

            return outputs4, outputs4_sub, feat_out16_4, feat_out32_4, KD_loss4
        else:
            return outputs4

    def forward(self, f_img, lbl=None,pos_id=None):
        if pos_id == 0:
            outputs = self.forward_path1(f_img)
        elif pos_id == 1:
            outputs = self.forward_path2(f_img)
        elif pos_id == 2:
            outputs = self.forward_path3(f_img)
        elif pos_id == 3:
            outputs = self.forward_path4(f_img)
        else:
            raise RuntimeError("Only Four Paths.")

        '''_, _, h, w = f_img.size()
        f_img = F.interpolate(f_img, (h+1, w+1), mode='bilinear', align_corners=True)
        T_logit_1234, T_logit_1, T_logit_2, T_logit_3, T_logit_4 = self.teacher(f_img)
        outputs = F.interpolate(T_logit_1234, (h, w), mode='bilinear', align_corners=True)'''

        if self.training:
            outputs_, outputs_sub, feat_out16, feat_out32, KD_loss = outputs
            loss = self.loss_fn(outputs_,lbl) +\
                    self.loss_fn(outputs_sub,lbl) +\
                    self.loss_fn(feat_out16,lbl) +\
                    self.loss_fn(feat_out32,lbl)+\
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
            if isinstance(child, (OhemCELoss2D,SegmentationLosses, BiSeNet, pspnet, nn.KLDivLoss)):
                continue
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, (FeatureFusionModule, BiSeNetOutput, Encoding, Attention, Layer_Norm)):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def pretrained_mp_load(self):
        if self.bise_path is not None:
            if os.path.isfile(self.bise_path):
                logger.info("Resuming training from pretrained '{}'".format(self.bise_path))
                print("Resuming training from pretrained '{}'".format(self.bise_path))
                model_state = torch.load(self.bise_path)["model_state"]
                self.load_state_dict(model_state, strict=False)
            else:
                logger.info("No pretrained found at '{}'".format(self.bise_path))

    def pretrained_init_1p(self):

        if self.bise_path is not None:
            if os.path.isfile(self.bise_path):
                logger.info("Resuming training from pretrained '{}'".format(self.bise_path))
                model_state = torch.load(self.bise_path)["model_state"]
                cp_state, ffm_state, conv_state1, conv_state2, conv16_aux_state, conv32_aux_state = split_1p_state_dict(model_state,self.path_num)

                self.cp1.load_state_dict(cp_state, strict=True)
                self.cp2.load_state_dict(cp_state, strict=True)
                self.cp3.load_state_dict(cp_state, strict=True)
                self.cp4.load_state_dict(cp_state, strict=True)

                self.ffm1.load_state_dict(ffm_state, strict=True)
                self.ffm2.load_state_dict(ffm_state, strict=True)
                self.ffm3.load_state_dict(ffm_state, strict=True)
                self.ffm4.load_state_dict(ffm_state, strict=True)

                self.conv_out_1.load_state_dict(conv_state1, strict=True)
                self.conv_out_2.load_state_dict(conv_state2, strict=True)
                self.conv_out_3.load_state_dict(conv_state1, strict=True)
                self.conv_out_4.load_state_dict(conv_state2, strict=True)


                self.conv_out16_1.load_state_dict(conv16_aux_state, strict=True)
                self.conv_out16_2.load_state_dict(conv16_aux_state, strict=True)
                self.conv_out16_3.load_state_dict(conv16_aux_state, strict=True)
                self.conv_out16_4.load_state_dict(conv16_aux_state, strict=True)

                self.conv_out32_1.load_state_dict(conv32_aux_state, strict=True)
                self.conv_out32_2.load_state_dict(conv32_aux_state, strict=True)
                self.conv_out32_3.load_state_dict(conv32_aux_state, strict=True)
                self.conv_out32_4.load_state_dict(conv32_aux_state, strict=True)
            else:
                logger.info("No pretrained found at '{}'".format(self.bise_path))

    def pretrained_init_2p(self):

        if self.bise_path is not None:
            if os.path.isfile(self.bise_path):
                logger.info("Resuming training from pretrained '{}'".format(self.bise_path))
                print("Resuming training from pretrained '{}'".format(self.bise_path))
                model_state = torch.load(self.bise_path)["model_state"]
                cp_s, ffm_s, enc_s, atn_s, ln_s, conv_out_s, conv16_aux_s, conv32_aux_s = split_2p_state_dict(model_state)

                self.cp1.load_state_dict(cp_s[0], strict=True)
                self.cp2.load_state_dict(cp_s[1], strict=True)
                self.cp3.load_state_dict(cp_s[0], strict=True)
                self.cp4.load_state_dict(cp_s[1], strict=True)

                self.ffm1.load_state_dict(ffm_s[0], strict=True)
                self.ffm2.load_state_dict(ffm_s[1], strict=True)
                self.ffm3.load_state_dict(ffm_s[0], strict=True)
                self.ffm4.load_state_dict(ffm_s[1], strict=True)

                self.enc1.load_state_dict(enc_s[0], strict=True)
                self.enc2.load_state_dict(enc_s[1], strict=True)
                self.enc3.load_state_dict(enc_s[0], strict=True)
                self.enc4.load_state_dict(enc_s[1], strict=True)

                self.atn1_2.load_state_dict(atn_s[0], strict=True)
                self.atn1_3.load_state_dict(atn_s[1], strict=True)
                self.atn1_4.load_state_dict(atn_s[0], strict=True)

                self.atn2_1.load_state_dict(atn_s[1], strict=True)
                self.atn2_3.load_state_dict(atn_s[1], strict=True)
                self.atn2_4.load_state_dict(atn_s[0], strict=True)

                self.atn3_1.load_state_dict(atn_s[1], strict=True)
                self.atn3_2.load_state_dict(atn_s[0], strict=True)
                self.atn3_4.load_state_dict(atn_s[0], strict=True)

                self.atn4_1.load_state_dict(atn_s[1], strict=True)
                self.atn4_2.load_state_dict(atn_s[0], strict=True)
                self.atn4_3.load_state_dict(atn_s[1], strict=True)

                self.layer_norm1.load_state_dict(ln_s[0], strict=True)
                self.layer_norm2.load_state_dict(ln_s[1], strict=True)
                self.layer_norm3.load_state_dict(ln_s[0], strict=True)
                self.layer_norm4.load_state_dict(ln_s[1], strict=True)

                self.conv_out_1.load_state_dict(conv_out_s[0], strict=True)
                self.conv_out_2.load_state_dict(conv_out_s[1], strict=True)
                self.conv_out_3.load_state_dict(conv_out_s[0], strict=True)
                self.conv_out_4.load_state_dict(conv_out_s[1], strict=True)

                self.conv_out16_1.load_state_dict(conv16_aux_s[0], strict=True)
                self.conv_out16_2.load_state_dict(conv16_aux_s[1], strict=True)
                self.conv_out16_3.load_state_dict(conv16_aux_s[0], strict=True)
                self.conv_out16_4.load_state_dict(conv16_aux_s[1], strict=True)

                self.conv_out32_1.load_state_dict(conv32_aux_s[0], strict=True)
                self.conv_out32_2.load_state_dict(conv32_aux_s[1], strict=True)
                self.conv_out32_3.load_state_dict(conv32_aux_s[0], strict=True)
                self.conv_out32_4.load_state_dict(conv32_aux_s[1], strict=True)

            else:
                logger.info("No pretrained found at '{}'".format(self.bise_path))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, norm_layer=None,*args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.norm_layer=norm_layer
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1,norm_layer=norm_layer)
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


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_layer=None, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.norm_layer=norm_layer
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1,norm_layer=norm_layer)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = norm_layer(out_chan, activation='none')
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, backbone, norm_layer=None):
        super(ContextPath, self).__init__()
        self.norm_layer=norm_layer
        self.backbone = backbone
        if backbone == 'resnet18':
            self.resnet = Resnet18(norm_layer=norm_layer)
            self.expansion = 1
        elif backbone == 'resnet34':
            self.resnet = Resnet34(norm_layer=norm_layer)
            self.expansion = 1
        elif backbone == 'resnet50':
            self.resnet = Resnet50(norm_layer=norm_layer)
            self.expansion = 4
        elif backbone == 'resnet101':
            self.resnet = Resnet101(norm_layer=norm_layer)
            self.expansion = 4
        elif backbone == 'resnet152':
            self.resnet = Resnet152(norm_layer=norm_layer)
            self.expansion = 4
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self.arm16 = AttentionRefinementModule(256*self.expansion, 128,norm_layer=norm_layer)
        self.arm32 = AttentionRefinementModule(512*self.expansion, 128,norm_layer=norm_layer)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1,norm_layer=norm_layer)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1,norm_layer=norm_layer)
        self.conv_avg = ConvBNReLU(512*self.expansion, 128, ks=1, stride=1, padding=0,norm_layer=norm_layer)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat4, feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8,feat16_up, feat32_up # x8, x16

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


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan,norm_layer=None, path_num=None, pid=None, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.path_num = path_num
        self.pid = pid
        self.norm_layer=norm_layer
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0,norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat

        n, c, h, w = feat_out.size()
        feat_out = feat_out[:, self.pid*c//self.path_num:(1+self.pid)*c//self.path_num]

        return feat_out

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


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=None, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.norm_layer=norm_layer
        self.bn = norm_layer(out_chan, activation='leaky_relu')
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


class Layer_Norm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super(Layer_Norm, self).__init__()
        self.ln = nn.GroupNorm(num_groups, num_channels)

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
