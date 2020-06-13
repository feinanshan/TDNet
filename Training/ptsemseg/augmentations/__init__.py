import logging

from ptsemseg.augmentations.augmentations import (
    RandomCrop,
    RandomHorizontallyFlip,
    RandomVerticallyFlip,
    Scale,
    RandomScale,
    RandomRotate,
    RandomTranslate,
    CenterCrop,
    Compose,
    ColorJitter,
    ColorNorm
)

logger = logging.getLogger("ptsemseg")

key2aug = {
    "rcrop": RandomCrop,
    "hflip": RandomHorizontallyFlip,
    "vflip": RandomVerticallyFlip,
    "scale": Scale,
    "rscale": RandomScale,
    "rotate": RandomRotate,
    "translate": RandomTranslate,
    "ccrop": CenterCrop,
    "colorjtr": ColorJitter,
    "colornorm": ColorNorm
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        logger.info("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](aug_param))
        logger.info("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)