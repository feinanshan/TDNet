import torch
import torch.nn as nn
import torch.nn.functional as F
import timeit
from .resnet import resnet18,resnet34,resnet50
import random
import pdb
import os
from .transformer import Encoding, Attention

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

class td4_psp18(nn.Module):
    """
    """
    def __init__(self,
                 nclass=21,
                 norm_layer=BatchNorm2d,
                 backbone='resnet18',
                 dilated=True,
                 aux=True,
                 multi_grid=True,
                 path_num=None,
                 model_path = None
                 ):
        super(td4_psp18, self).__init__()

        self.psp_path = model_path
        self.path_num = path_num
        self.norm_layer = norm_layer
        self._up_kwargs = up_kwargs
        self.nclass = nclass

        # copying modules from pretrained models
        self.backbone = backbone
        assert(backbone == 'resnet50' or backbone == 'resnet34' or backbone == 'resnet18')
        assert(path_num == 4)

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

        self.pretrained1 = ResNet_(dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        self.pretrained2 = ResNet_(dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        self.pretrained3 = ResNet_(dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        self.pretrained4 = ResNet_(dilated=dilated, multi_grid=multi_grid,
                                           deep_base=deep_base, norm_layer=norm_layer)
        # bilinear upsample options

        self.psp1 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=0)
        self.psp2 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=1)
        self.psp3 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=0)
        self.psp4 =  PyramidPooling(512*self.expansion, norm_layer, self._up_kwargs, path_num=self.path_num//2, pid=1)

        self.enc1 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc2 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc3 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)
        self.enc4 = Encoding(512*self.expansion,64,512*self.expansion,norm_layer)

 
        self.atn1_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_3 = Attention(512*self.expansion,64,norm_layer)
        self.atn1_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn2_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn2_3 = Attention(512*self.expansion,64,norm_layer)
        self.atn2_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn3_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn3_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn3_4 = Attention(512*self.expansion,64,norm_layer)
        
        self.atn4_1 = Attention(512*self.expansion,64,norm_layer)
        self.atn4_2 = Attention(512*self.expansion,64,norm_layer)
        self.atn4_3 = Attention(512*self.expansion,64,norm_layer)
        
        self.layer_norm1 = Layer_Norm([97,193])
        self.layer_norm2 = Layer_Norm([97,193])
        self.layer_norm3 = Layer_Norm([97,193])
        self.layer_norm4 = Layer_Norm([97,193])

        self.head1 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head2 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head3 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
        self.head4 = FCNHead(512*self.expansion*1, nclass, norm_layer, chn_down=4)
            
        self.pretrained_mp_load()
        self.Q_queue = []
        self.V_queue = []
        self.K_queue = []


    def buffer_contral(self,q,k,v):
        assert(len(self.Q_queue)==len(self.V_queue))
        assert(len(self.Q_queue)==len(self.K_queue))

        self.Q_queue.append(q)
        self.V_queue.append(v)
        self.K_queue.append(k)
        
        if len(self.Q_queue)>3:
            self.Q_queue.pop(0)
            self.V_queue.pop(0)
            self.K_queue.pop(0)
                    

    def forward_path1(self, img):
        z1 = self.psp1(self.pretrained1(img))

        q_cur,v_cur = self.enc1(z1, pre=False) 

        if len(self.Q_queue)<3: 
            output = self.head1(self.layer_norm1(v_cur))
        else:
            v_2_ = self.atn1_2(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
            v_3_ = self.atn1_3(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
            v_4_ = self.atn1_4(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z1.size())
            
            #pdb.set_trace()
 
            output = self.head1(self.layer_norm1(v_4_ + v_cur))

        q_cur,k_cur,v_cur = self.enc1(z1, pre=True) 
        self.buffer_contral(q_cur,k_cur,v_cur)
        return output


    def forward_path2(self, img):
        z2 = self.psp2(self.pretrained2(img))

        q_cur,v_cur = self.enc2(z2, pre=False) 

        if len(self.Q_queue)<3: 
            output = self.head2(self.layer_norm2(v_cur))
        else:            
            v_2_ = self.atn2_3(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
            v_3_ = self.atn2_4(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
            v_4_ = self.atn2_1(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z2.size())
 
            output = self.head2(self.layer_norm2(v_4_ + v_cur))

        q_cur,k_cur,v_cur = self.enc2(z2, pre=True) 
        self.buffer_contral(q_cur,k_cur,v_cur)
        return output


    def forward_path3(self, img):
        z3 = self.psp3(self.pretrained3(img))

        q_cur,v_cur = self.enc3(z3, pre=False) 

        if len(self.Q_queue)<3: 
            output = self.head3(self.layer_norm3(v_cur))
        else:            
            v_2_ = self.atn3_4(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
            v_3_ = self.atn3_1(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
            v_4_ = self.atn3_2(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z3.size())
 
            output = self.head3(self.layer_norm3(v_4_ + v_cur))

        q_cur,k_cur,v_cur = self.enc3(z3, pre=True)
        self.buffer_contral(q_cur,k_cur,v_cur)
        return output


    def forward_path4(self, img):
        z4 = self.psp4(self.pretrained4(img))

        q_cur,v_cur = self.enc4(z4, pre=False) 

        if len(self.Q_queue)<3: 
            output = self.head4(self.layer_norm4(v_cur))
        else:            
            v_2_ = self.atn4_1(self.K_queue[0],      self.V_queue[0], self.Q_queue[1], fea_size=None)
            v_3_ = self.atn4_2(self.K_queue[1], v_2_+self.V_queue[1], self.Q_queue[2], fea_size=None)
            v_4_ = self.atn4_3(self.K_queue[2], v_3_+self.V_queue[2], q_cur, fea_size=z4.size())
 
            output = self.head4(self.layer_norm4(v_4_ + v_cur))

        q_cur,k_cur,v_cur = self.enc4(z4, pre=True)
        self.buffer_contral(q_cur,k_cur,v_cur)
        return output



    def forward(self, img, pos_id=0):
        _, _, h, w = img.size()
        if pos_id == 0:
            output = self.forward_path1(img)
        elif pos_id == 1:
            output = self.forward_path2(img)
        elif pos_id == 2:
            output = self.forward_path3(img)
        elif pos_id == 3:
            output = self.forward_path4(img)
            
        output = F.interpolate(output, (h, w), **self._up_kwargs)

        return output
        
        
    def pretrained_mp_load(self):
        if self.psp_path is not None:
            if os.path.isfile(self.psp_path):
                print("Loading pretrained model from '{}'".format(self.psp_path))
                model_state = torch.load(self.psp_path)
                self.load_state_dict(model_state, strict=True)

            else:
                print("No pretrained found at '{}'".format(self.psp_path))


class PyramidPooling(nn.Module):

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

    def forward(self, x):
        return self.conv5(x)



class Layer_Norm(nn.Module):
    def __init__(self, shape):
        super(Layer_Norm, self).__init__()
        self.ln = nn.LayerNorm(shape)

    def forward(self, x):
        return self.ln(x)

