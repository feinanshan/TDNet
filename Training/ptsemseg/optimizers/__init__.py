import logging
import copy

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop
from .adaoptimizer import AdaOptimizer
logger = logging.getLogger("ptsemseg")

key2opt = {
    "sgd": SGD,
    "adam": Adam,
    "asgd": ASGD,
    "adamax": Adamax,
    "adadelta": Adadelta,
    "adagrad": Adagrad,
    "rmsprop": RMSprop,
    "adaoptimizer": AdaOptimizer,
}


def get_optimizer(cfg, model):
    if cfg["optimizer"] is None:
        logger.info("Using SGD optimizer")
        return SGD(model.parameters())

    else:
        opt_name = cfg["optimizer"]["name"]
        if opt_name not in key2opt:
            raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

        optimizer_cls = key2opt[opt_name]
        if opt_name == "adaoptimizer":
            param_dict = copy.deepcopy(cfg["optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model, **param_dict) # module for multi GPU

        else:
            param_dict = copy.deepcopy(cfg["optimizer"])
            param_dict.pop("name")
            optimizer = optimizer_cls(model.parameters(), **param_dict)

        logger.info("Using {} optimizer".format(opt_name))
        return optimizer
