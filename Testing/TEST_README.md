# Segmenting Video Sequences with TDNet

### 1. Prepare Data
Save video frames with their IDs as names. An examplar video is saved in `./data/vid1`.

### 2. Prepare Model:
* Download pretrained models from the [[G-cloud](https://drive.google.com/file/d/1TgoAHhbULe14cUQF-zbo7KudSK-CJDga/view?usp=sharing)].
* Decompress `checkpoint.zip` and put the folder into this dirctory.

### 3. Running:

Test with TD2-PSP50: 
```bash
python test.py --gpu 1 --model td2-psp50
```

Test with TD4-PSP18:
```bash
python test.py --gpu 1 --model td4-psp18
```

For performance comparison, you can also run with PSPNet-101: 
```bash
python test.py --gpu 1 --model psp101
```

### 4. Latency/Speed 

|Model         |Latency (on Titan Xp)  |Per-frame FLOPs  |
|:------------:|:-----------------------:|:-----------------:|
|PSPNet101     |~360ms/f               |~1.2T/f           |
|TD2-PSP50     | ~180ms/f              |~541G/f           |
|TD4-PSP18     |~85ms/f                |~239G/f           |
