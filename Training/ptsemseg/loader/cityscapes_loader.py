import os
import torch
import numpy as np
import imageio

from torch.utils import data
import random
from ptsemseg.utils import recursive_glob
from ptsemseg.augmentations import Compose, RandomHorizontallyFlip, RandomRotate, Scale


class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))



    def __init__(
        self,
        root,
        split="train",
        augmentations=None,
        test_mode=False,
        model_name=None,
        interval=2,
        path_num=2,
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.path_num= path_num
        self.interval = interval
        self.root = root
        self.split = split
        self.augmentations = augmentations
        self.test_mode=test_mode
        self.model_name=model_name
        self.n_classes = 19
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.videos_base = os.path.join(self.root, "leftImg8bit_sequence", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        if not self.test_mode:
            img_path = self.files[self.split][index].rstrip()
            lbl_path = os.path.join(
                self.annotations_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
            )
            lbl = imageio.imread(lbl_path)
            lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

            vid_info = img_path.split('/')[-1].split('_')
            city, seq, cur_frame = vid_info[0], vid_info[1], vid_info[2]
            f4_id = int(cur_frame)
            f3_id = f4_id - random.randint(1, self.interval)
            f2_id = f3_id - random.randint(1, self.interval)
            f1_id = f2_id - random.randint(1, self.interval)

            f4_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, f4_id)))
            f4_img = imageio.imread(f4_path)
            f4_img = np.array(f4_img, dtype=np.uint8)

            f3_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, f3_id)))
            f3_img = imageio.imread(f3_path)
            f3_img = np.array(f3_img, dtype=np.uint8)
            
            f2_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, f2_id)))
            f2_img = imageio.imread(f2_path)
            f2_img = np.array(f2_img, dtype=np.uint8)

            f1_path = os.path.join(self.videos_base, city, ("%s_%s_%06d_leftImg8bit.png" % (city, seq, f1_id)))
            f1_img = imageio.imread(f1_path)
            f1_img = np.array(f1_img, dtype=np.uint8)

            if self.augmentations is not None:
                [f4_img, f3_img, f2_img, f1_img], lbl = self.augmentations([f4_img, f3_img, f2_img, f1_img], lbl)

            f4_img = f4_img.float()
            f3_img = f3_img.float()
            f2_img = f2_img.float()
            f1_img = f1_img.float()
            lbl = torch.from_numpy(lbl).long()

            if self.path_num == 4:
                return [f1_img, f2_img, f3_img, f4_img], lbl
            else:
                return [f3_img, f4_img], lbl


    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def decode_pred(self, mask):
        # Put all void classes to zero
        for _predc in range(self.n_classes):
            mask[mask == _predc] = self.valid_classes[_predc]
        return mask.astype(np.uint8)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip(0.5)])

    local_path = "/datasets01/cityscapes/112817/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data_samples in enumerate(trainloader):
        imgs, labels = data_samples
        import pdb

        pdb.set_trace()
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = input()
        if a == "ex":
            break
        else:
            plt.close()
