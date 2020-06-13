import logging
import torch
import functools

from ptsemseg.loss.loss import (
    SegmentationLosses,
    OhemCELoss2D,
)


logger = logging.getLogger("ptsemseg")

key2loss = {
    "SegmentationLosses": SegmentationLosses,
    "OhemCELoss2D": OhemCELoss2D,
}


def get_loss_function(cfg):
    assert(cfg["loss"] is not None)
    loss_dict = cfg["loss"]
    loss_name = loss_dict["name"]
    loss_params = {k: v for k, v in loss_dict.items() if k != "name"}
    if loss_name not in key2loss:
        raise NotImplementedError("Loss {} not implemented".format(loss_name))

    if loss_name == "OhemCELoss2D":
        n_img_per_gpu = int(cfg["batch_size"]/torch.cuda.device_count())
        cropsize = cfg["train_augmentations"]["rcrop"]
        n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
        loss_params["n_min"] = n_min

    logger.info("Using {} with {} params".format(loss_name, loss_params))
    return key2loss[loss_name](**loss_params)
