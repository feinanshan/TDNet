# TDNet
Temporally Distributed Networks for Fast Video Semantic Segmentation (CVPR'20)

[Ping Hu](http://cs-people.bu.edu/pinghu/), [Fabian Caba Heilbron](http://fabiancaba.com/), [Oliver Wang](http://www.oliverwang.info/), [Zhe Lin](http://sites.google.com/site/zhelin625/), [Stan Sclaroff](http://www.cs.bu.edu/~sclaroff/), [Federico Perazzi](https://fperazzi.github.io/)

[[Paper Link](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Temporally_Distributed_Networks_for_Fast_Video_Semantic_Segmentation_CVPR_2020_paper.pdf)] [[Project Page](http://cs-people.bu.edu/pinghu/TDNet.html)]


We present TDNet, a temporally distributed network designed for fast and accurate video semantic segmentation. We observe that features extracted from a certain high-level layer of a deep CNN can be approximated by composing features extracted from several shallower subnetworks. Leveraging the inherent temporal continuity in videos, we distribute these sub-networks over sequential frames. Therefore, at each time step, we only need to perform a lightweight computation to extract a sub-features group from a single sub-network. The full features used for segmentation are then recomposed by the application of a novel attention propagation module that compensates for geometry deformation between frames. A grouped knowledge distillation loss is also introduced to further improve the representation power at both full and sub-feature levels. Experiments on Cityscapes, CamVid, and NYUD-v2 demonstrate that our method achieves state-of-the-art accuracy with significantly faster speed and lower latency


## Installation:

#### Requirements:
1. Linux
2. Python 3.7
3. Pytorch 1.1.0
4. NVIDIA GPU + CUDA 10.0

#### Build

```bash
pip install -r requirements.txt
```

## Test with TDNet

see [TEST_README.md](./Testing/TEST_README.md)

## Train with TDNet

see [TRAIN_README.md](./Training/TRAIN_README.md)


## Citation
If you find TDNet is helpful in your research, please consider citing:

    @InProceedings{hu2020tdnet,
    title={Temporally Distributed Networks for Fast Video Semantic Segmentation},
    author={Hu, Ping and Caba, Fabian and Wang, Oliver and Lin, Zhe and Sclaroff, Stan and Perazzi, Federico},
    journal={CVPR},
    year={2020}
    }

## Disclaimer

This is a pytorch re-implementation of TDNet, please refer to the original paper [Temporally Distributed Networks for Fast Video Semantic Segmentation](http://openaccess.thecvf.com/content_CVPR_2020/papers/Hu_Temporally_Distributed_Networks_for_Fast_Video_Semantic_Segmentation_CVPR_2020_paper.pdf) for comparisons.

## References

- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg)

