## Training for TDNet

### 1. Prepare Data
Download and save datasets. 

* [Cityscapes](https://www.cityscapes-dataset.com/): put `leftImg8bit_sequence` and `gtFine` in the same folder.
* [Camvid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [NYUDepthV2 (~14GB)](https://drive.google.com/file/d/1afnlZoCS7FUzXeQq_UzUdkHB2vmV1jEo/view?usp=sharing)

### 2. Download Models

Download the [pretained models](https://drive.google.com/file/d/14udr_GoNdFknDghjXJApL0hElC1BoVD-/view?usp=sharing) for subnetworks, and put them into `./pretrained`. 

* td2_fa18:  takes two [FANet-18](https://arxiv.org/pdf/2007.03815.pdf) as subnetworks.
* td2_psp50: takes two PSPNet-50 as subnetworks.
* td4_psp18: takes four PSPNet-18 as subnetworks.

### 3. Modify Codes
Modify `*.yml` files in `./config`
* ''data:path'': path to dataset 
* ''training:batch_size'': batch_size
* ''training:train_augmentations:rcrop'': input size for training

### 4. Training
Run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --config configs/*.yml
```


### **Notice
Still working on this implementation.
