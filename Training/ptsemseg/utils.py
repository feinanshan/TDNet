"""
Misc Utility functions
"""
import os
import torch
import logging
import datetime
import numpy as np

from collections import OrderedDict


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def alpha_blend(input_image, segmentation_mask, alpha=0.5):
    """Alpha Blending utility to overlay RGB masks on RBG images
        :param input_image is a np.ndarray with 3 channels
        :param segmentation_mask is a np.ndarray with 3 channels
        :param alpha is a float value
    """
    blended = np.zeros(input_image.size, dtype=np.float32)
    blended = input_image * alpha + segmentation_mask * (1 - alpha)
    return blended

def split_fanet_dict(state_dict, path_num =None):
    backbone_state = OrderedDict()
    ffm_32_state = OrderedDict()
    ffm_16_state = OrderedDict()
    ffm_8_state = OrderedDict()
    ffm_4_state = OrderedDict()
    output_state = OrderedDict()
    output_aux_state = OrderedDict()

    for k, v in state_dict.items():
        s_k = k.split('.')
        if s_k[0] == 'resnet':
            backbone_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'ffm_32':
            ffm_32_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'ffm_16':
            ffm_16_state['.'.join(s_k[1:])] = v
            
        if s_k[0] == 'ffm_8':
            ffm_8_state['.'.join(s_k[1:])] = v
            
        if s_k[0] == 'ffm_4':
            ffm_4_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'clslayer_8':
            output_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'clslayer_32':
            output_aux_state['.'.join(s_k[1:])] = v

    return backbone_state, ffm_32_state, ffm_16_state, ffm_8_state, ffm_4_state, output_state, output_aux_state


def split_psp_dict(state_dict, path_num =None):
    """Split a PSPNet model into different part
       :param state_dict is the loaded DataParallel model_state
    """
    model_state = convert_state_dict(state_dict)

    backbone_state = OrderedDict()
    psp_state = OrderedDict()
    head_state1 = OrderedDict()
    head_state2 = OrderedDict()
    head_state3 = OrderedDict()
    head_state4 = OrderedDict()
    auxlayer_state = OrderedDict()

    for k, v in model_state.items():
        s_k = k.split('.')
        if s_k[0] == 'pretrained':
            backbone_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'head':
            pk = s_k[1:]
            if pk[1] == '0':
                psp_state['.'.join(pk[2:])] = v
            else:
                pk[1] = str(int(pk[1]) - 1)
                if pk[1] == '0':  #Shift channel params
                    o_c, i_c, h_, w_ = v.size()
                    shifted_param_l = []
                    step1 = i_c//2//path_num
                    step2 = i_c//8//path_num
                    for id in range(path_num):
                        idx1 = range(id*step1,id*step1+step1)
                        idx2 = range(i_c*4//8+id*step2,i_c*4//8+id*step2+step2)
                        idx3 = range(i_c*5//8+id*step2,i_c*5//8+id*step2+step2)
                        idx4 = range(i_c*6//8+id*step2,i_c*6//8+id*step2+step2)
                        idx5 = range(i_c*7//8+id*step2,i_c*7//8+id*step2+step2)
                        shifted_param_l.append(v[:,idx1,:,:])
                        shifted_param_l.append(v[:,idx2,:,:])
                        shifted_param_l.append(v[:,idx3,:,:])
                        shifted_param_l.append(v[:,idx4,:,:])
                        shifted_param_l.append(v[:,idx5,:,:])
                    v1 = torch.cat(shifted_param_l[:5], dim=1)
                    v2 = torch.cat(shifted_param_l[5:10], dim=1)
                    if path_num==2:
                        v3 = torch.cat(shifted_param_l[:5], dim=1)
                        v4 = torch.cat(shifted_param_l[5:10], dim=1)
                    elif path_num==4:
                        v3 = torch.cat(shifted_param_l[10:15], dim=1)
                        v4 = torch.cat(shifted_param_l[15:20], dim=1)
                    else:
                        raise RuntimeError("Only support 2 or 4 path")

                    head_state1['.'.join(pk)] =v1
                    head_state2['.'.join(pk)] =v2
                    head_state3['.'.join(pk)] =v3
                    head_state4['.'.join(pk)] =v4
                else:
                    head_state1['.'.join(pk)] = v
                    head_state2['.'.join(pk)] = v
                    head_state3['.'.join(pk)] = v
                    head_state4['.'.join(pk)] = v

        if s_k[0] == 'auxlayer':
            auxlayer_state['.'.join(s_k[1:])] = v

    return backbone_state, psp_state, head_state1, head_state2, head_state3, head_state4, auxlayer_state


def split_psp_state_dict(state_dict, path_num = 4):
    """Split a PSPNet model into different part
       :param state_dict is the loaded DataParallel model_state
    """
    backbone_state = OrderedDict()
    psp_state = OrderedDict()
    grp_state1 = OrderedDict()
    grp_state2 = OrderedDict()
    grp_state3 = OrderedDict()
    grp_state4 = OrderedDict()
    head_state = OrderedDict()
    auxlayer_state = OrderedDict()

    for k, v in state_dict.items():
        
        s_k = k.split('.')
        if s_k[0] == 'pretrained':
            backbone_state['.'.join(s_k[1:])] = v

        if s_k[0] == 'head':
            pk = s_k[1:]
            if pk[1] == '0':
                psp_state['.'.join(pk[2:])] = v
            else:
                pk[1] = str(int(pk[1]) - 1)
                if pk[1] == '0':  #Shift channel params
                    o_c, i_c, h_, w_ = v.size()
                    shifted_param_l = []
                    step1 = i_c//2//path_num
                    step2 = i_c//8//path_num
                    for id in range(path_num):
                        idx1 = range(id*step1,id*step1+step1)
                        idx2 = range(i_c*4//8+id*step2,i_c*4//8+id*step2+step2)
                        idx3 = range(i_c*5//8+id*step2,i_c*5//8+id*step2+step2)
                        idx4 = range(i_c*6//8+id*step2,i_c*6//8+id*step2+step2)
                        idx5 = range(i_c*7//8+id*step2,i_c*7//8+id*step2+step2)
                        shifted_param_l.append(v[:,idx1,:,:])
                        shifted_param_l.append(v[:,idx2,:,:])
                        shifted_param_l.append(v[:,idx3,:,:])
                        shifted_param_l.append(v[:,idx4,:,:])
                        shifted_param_l.append(v[:,idx5,:,:])
                    v1 = torch.cat(shifted_param_l[:5], dim=1)
                    v2 = torch.cat(shifted_param_l[5:10], dim=1)

                    if path_num==2:
                        v3 = torch.cat(shifted_param_l[:5], dim=1)
                        v4 = torch.cat(shifted_param_l[5:10], dim=1)
                    elif path_num==4:
                        v3 = torch.cat(shifted_param_l[10:15], dim=1)
                        v4 = torch.cat(shifted_param_l[15:20], dim=1)
                    else:
                        raise RuntimeError("Only support 2 or 4 path")
                        
                    grp_state1['.'.join(pk)] =v1
                    grp_state2['.'.join(pk)] =v2
                    grp_state3['.'.join(pk)] =v3
                    grp_state4['.'.join(pk)] =v4
                else:
                    pk[1] = str(int(pk[1]) - 1)
                    head_state['.'.join(pk)] =v

        if s_k[0] == 'auxlayer':
            auxlayer_state['.'.join(s_k[1:])] = v

    return backbone_state, psp_state, grp_state1, grp_state2, grp_state3, grp_state4, head_state, auxlayer_state

def clean_state_dict(state_dict, key=None):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.split('.')[0] != key:  # remove `module.`
            new_state_dict[k] = v
    return new_state_dict

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_logger(logdir):
    logger = logging.getLogger("ptsemseg")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
