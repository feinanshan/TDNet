import math
import numbers
import random
import torchvision.transforms.functional as tf
from PIL import Image, ImageOps
import PIL.ImageEnhance as ImageEnhance
import torchvision.transforms as transforms
import numpy as np

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, imgs, mask=None):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for img in imgs:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img, mode="RGB")
                if mask is not None:
                    mask_ = Image.fromarray(mask, mode="L")
            imgs_.append(img)
            self.PIL2Numpy = True
            if mask is not None:
                assert img.size == mask_.size

        for a in self.augmentations:
            imgs_, mask_ = a(imgs_, mask_)
        return imgs_, mask_


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            mask_ = mask
            if self.padding > 0:
                img = ImageOps.expand(img, border=self.padding, fill=0)
                mask_ = ImageOps.expand(mask_, border=self.padding, fill=0)

            assert img.size == mask_.size
            w, h = img.size
            th, tw = self.size
            if w == tw and h == th:
                imgs_.append(img)
            else:
                if w < tw or h < th:
                    img = img.resize((tw, th), Image.BILINEAR)
                    mask_ = mask_.resize((tw, th), Image.NEAREST)
                else:
                    if idx==0:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                    img = img.crop((x1, y1, x1 + tw, y1 + th))
                    mask_ = mask_.crop((x1, y1, x1 + tw, y1 + th))
                imgs_.append(img)
        return imgs_,mask_


class ColorJitter(object):
    def __init__(self, p):
        brightness = p[0]
        contrast =  p[1]
        saturation = p[2]

        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            if idx==0:
                r_brightness = random.uniform(self.brightness[0], self.brightness[1])
                r_contrast = random.uniform(self.contrast[0], self.contrast[1])
                r_saturation = random.uniform(self.saturation[0], self.saturation[1])
            img = ImageEnhance.Brightness(img).enhance(r_brightness)
            img = ImageEnhance.Contrast(img).enhance(r_contrast)
            img = ImageEnhance.Color(img).enhance(r_saturation)
            imgs_.append(img)
        return imgs_, mask



class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            w, h = img.size
            th, tw = self.size
            x1 = int(round((w - tw) / 2.0))
            y1 = int(round((h - th) / 2.0))
            img = img.crop((x1, y1, x1 + tw, y1 + th))
            mask_ = mask.crop((x1, y1, x1 + tw, y1 + th))
            imgs_.append(img)
        return imgs_, mask_


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, mask):

        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            mask_ = mask
            if idx==0:
                pro = random.random()
            if pro < self.p:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                mask_ = mask_.transpose(Image.FLIP_LEFT_RIGHT)
            imgs_.append(img)

        return imgs_, mask_


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, imgs, mask):

        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            if idx==0:
                pro = random.random()
            if pro < self.p:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                mask_ = mask.transpose(Image.FLIP_TOP_BOTTOM)
            imgs_.append(img)

        return imgs_, mask_

class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            img = img.resize(self.size, Image.BILINEAR)
            mask_ = mask.resize(self.size, Image.NEAREST)
            imgs_.append(img)
        return imgs_, mask_



class RandomTranslate(object):
    def __init__(self, offset):
        # tuple (delta_x, delta_y)
        self.offset = offset

    def __call__(self, imgs, mask):

        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            if idx==0:
                x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
                y_offset = int(2 * (random.random() - 0.5) * self.offset[1])

            x_crop_offset = x_offset
            y_crop_offset = y_offset
            if x_offset < 0:
                x_crop_offset = 0
            if y_offset < 0:
                y_crop_offset = 0

            cropped_img = tf.crop(
                img,
                y_crop_offset,
                x_crop_offset,
                img.size[1] - abs(y_offset),
                img.size[0] - abs(x_offset),
            )

            if x_offset >= 0 and y_offset >= 0:
                padding_tuple = (0, 0, x_offset, y_offset)

            elif x_offset >= 0 and y_offset < 0:
                padding_tuple = (0, abs(y_offset), x_offset, 0)

            elif x_offset < 0 and y_offset >= 0:
                padding_tuple = (abs(x_offset), 0, 0, y_offset)

            elif x_offset < 0 and y_offset < 0:
                padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)

            img = tf.pad(cropped_img, padding_tuple, padding_mode="reflect")
            mask_ = tf.affine(
                    mask,
                    translate=(-x_offset, -y_offset),
                    scale=1.0,
                    angle=0.0,
                    shear=0.0,
                    fillcolor=250,
                )
            imgs_.append(img)
        return imgs_, mask_


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            if idx == 0:
                rotate_degree = random.random() * 2 * self.degree - self.degree
            img = tf.affine(
                    img,
                    translate=(0, 0),
                    scale=1.0,
                    angle=rotate_degree,
                    resample=Image.BILINEAR,
                    fillcolor=(0, 0, 0),
                    shear=0.0,
                )
            mask_ = tf.affine(
                    mask,
                    translate=(0, 0),
                    scale=1.0,
                    angle=rotate_degree,
                    resample=Image.NEAREST,
                    fillcolor=250,
                    shear=0.0,
                )
            imgs_.append(img)
        return imgs_, mask_



class RandomScale(object):
    def __init__(self, scales=(1, )):
        self.scales = scales

    def __call__(self, imgs, mask):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            assert img.size == mask.size
            W, H = img.size
            if idx==0:
                scale = random.choice(self.scales)
            w, h = int(W * scale), int(H * scale)
            img = img.resize((w, h), Image.BILINEAR)
            mask_ = mask.resize((w, h), Image.NEAREST)
            imgs_.append(img)
        return imgs_, mask_


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs, mask=None):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            mask_ = mask
            if mask_ is not None:
                assert img.size == mask_.size
                mask_ = mask_.resize((self.size[1],self.size[0]), Image.NEAREST)
            img = img.resize((self.size[1],self.size[0]), Image.BILINEAR)

            imgs_.append(img)
        return imgs_, mask_

class ColorNorm(object):
    def __init__(self, mean_std):
        mean = tuple(mean_std[0])
        std = tuple(mean_std[1])
        self.to_tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean,std)

    def __call__(self, imgs, mask=None):
        assert ( isinstance(imgs, list))
        imgs_ = []
        for (idx, img) in enumerate(imgs):
            img = self.norm(self.to_tensor(img))
            mask_ = np.array(mask).astype(np.int64)
            imgs_.append(img)
        return imgs_, mask_