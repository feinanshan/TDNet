''' Define the sublayers in encoder/decoder layer '''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class Encoding(nn.Module):
    def __init__(self, d_model, d_k, d_v, norm_layer=None, dropout=0.1):
        super(Encoding, self).__init__()

        self.norm_layer = norm_layer

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Sequential(ConvBNReLU(d_model, d_k, ks=1, stride=1, padding=0, norm_layer=norm_layer),
                                  ConvBNReLU(d_k, d_k, ks=1, stride=1, padding=0, norm_layer=None))

        self.w_ks = nn.Sequential(ConvBNReLU(d_model, d_k, ks=1, stride=1, padding=0, norm_layer=norm_layer),
                                  ConvBNReLU(d_k, d_k, ks=1, stride=1, padding=0, norm_layer=None))

        self.w_vs = nn.Sequential(ConvBNReLU(d_model, d_v, ks=1, stride=1, padding=0, norm_layer=None))

        self.maxpool = nn.MaxPool2d(kernel_size=1, stride=3, padding=0)

    def forward(self, fea, pre=None, start=None):

        n_,c_,h_,w_ = fea.size()

        d_k, d_v = self.d_k, self.d_v

        if pre:

            fea = self.maxpool(fea)

            n_,c_,h_,w_ = fea.size()

            k_ = self.w_ks(fea).view(n_, d_k, h_, w_)
            v_ = self.w_vs(fea).view(n_, d_v, h_, w_)

            k_ = k_.permute(0, 2, 3, 1).contiguous().view(n_, -1, d_k)  # n x (*h*w) x c
            v_ = v_.permute(0, 2, 3, 1).contiguous().view(n_, -1, d_v)  # n x (*h*w) x c
            if start:
                return k_, v_, None
            else:
                q_ = self.w_qs(fea).view(n_, d_k, h_, w_)
                q_ = q_.permute(0, 2, 3, 1).contiguous().view(n_, -1, d_k)  # n x (*h*w) x c
                return k_, v_, q_

        else:
            v = self.w_vs(fea).view(n_, d_v, h_, w_)
            q = self.w_qs(fea).view(n_, d_k, h_, w_)
            q = q.permute(0, 2, 3, 1).contiguous().view(n_, -1, d_k)  # n x (*h*w) x c
            return v, q

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


class Attention(nn.Module):
    def __init__(self, d_v, d_k, norm_layer=None, dropout=0.1):
        super(Attention,self).__init__()

        self.norm_layer = norm_layer
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))

        self.fc = nn.Sequential(ConvBNReLU(d_v, d_v, ks=1, stride=1, padding=0, norm_layer=None))

        self.dropout = nn.Dropout(dropout)

    def forward(self, k_src, v_src, q_tgr, mask=None,fea_size=None):
        '''
        :param k_src: key of previous frame
        :param v_src: value of previous frame
        :param q_tgr: query of current frame
        :param mask:  attention range
        :return: aggregated feature
        '''
        mask = None
        output = self.attention(q_tgr, k_src, v_src, mask=mask)

        N,P,C = output.size()

        output = output.view(-1,C).view(N*P,C,1,1)
        output = self.dropout(self.fc(output))
        output = output.view(N*P,C).view(N,P,C)

        if fea_size is not None:
            n, c, h, w = fea_size
            output = output.permute(0,2,1).contiguous().view(n, -1, h, w)

        return output

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


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward,self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention,self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)

        output = torch.bmm(attn, v)

        return output


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, norm_layer=None, bias=True, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = bias)
        self.norm_layer = norm_layer
        if norm_layer is not None:
            self.bn = norm_layer(out_chan, activation='leaky_relu')
  
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        
        if self.norm_layer is not None:
            x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)