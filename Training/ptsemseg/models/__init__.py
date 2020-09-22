import copy
import torchvision.models as models

from encoding.nn import SyncBatchNorm
from ptsemseg.models.td4_psp.pspnet_4p import pspnet_4p
from ptsemseg.models.td4_psp.td4_psp import td4_psp
from ptsemseg.models.td2_psp.pspnet_2p import pspnet_2p
from ptsemseg.models.td2_psp.td2_psp import td2_psp
from ptsemseg.models.td2_fanet.td2_fa import td2_fa

def get_model(model_dict, nclass, loss_fn=None, mdl_path=None, teacher=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if loss_fn is not  None:
        param_dict['loss_fn'] = loss_fn

    if mdl_path is not  None:
        param_dict['mdl_path'] = mdl_path

    if teacher is not  None:
        param_dict['teacher'] = teacher

    if param_dict['syncBN']:
        param_dict['norm_layer'] = SyncBatchNorm
    param_dict.pop('syncBN')
    
    model = model(nclass=nclass, **param_dict)
    return model


def _get_model_instance(name):
    try:
        return {
            "pspnet_4p": pspnet_4p,
            "td4_psp": td4_psp,
            "pspnet_2p": pspnet_2p,
            "td2_psp": td2_psp,
            "td2_fa": td2_fa,
        }[name]
    except:
        raise ("Model {} not available".format(name))
