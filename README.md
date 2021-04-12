# scan2cap-2d

## Setup

### Step 1 - Download & Unzip the Required Files
* Download the `color` and `instance` renders for one of the following experiments:
1. Annotated Viewpoints:
    * [annotated_viewpoints_color.zip](https://www.google.com)
    * [annotated_viewpoints_instance.zip](https://www.google.com)

2. Estimated Viewpoints (3D-2D Backprojected)
    * [annotated_viewpoints_color.zip](https://www.google.com)
    * [annotated_viewpoints_instance.zip](https://www.google.com)

3. Bird's Eye Viewpoints (Top-Down)
    * [annotated_viewpoints_color.zip](https://www.google.com)
    * [annotated_viewpoints_instance.zip](https://www.google.com)

* Download the ScanRefer `train` and `validation` splits:
    * [scanrefer.zip](https://www.google.com)


and unzip the downloaded files to your desired location.

--- 
### Step 2 - Set Proper Paths
Set the following paths in `scan2cap-2d/lib/conf.py` based on your needs:

```
CONF.PATH.DATA_ROOT = '/home/user/data'
CONF.PATH.CODE_ROOT = '/home/user/code'
CONF.PATH.SCANNET_DIR = "/scannet/public/v2"
```
---

### Step 3 - Run!

    ap.add_argument("--exp_type", default="nret", help="retrieval or nonretrieval")
    ap.add_argument("--dataset", default="scanrefer", help="scanrefer or referit")
    ap.add_argument("--viewpoint", default="annotated", help="annotated, estimated or bev")
    ap.add_argument("--box", default="oracle", help="oracle, mrcnn or votenet")

1. Preprocessing
```
python main.py --prep --exp_type $ET --dataset $DS --viewpoint $VP --box $BX
```
where variables can take the following permutations:

| $DS           | $VP  |  $BX  | Comments
|:-----| :-----| :-----|:-----|
| scanrefer | annotated| oracle | Extracts oracle bounding boxes, bounding box features and global features from annotated viewpoints.
| scanrefer | annotated| mrcnn | Extracts MaskRCNN detected bounding boxes, bounding box features and global features from annotated viewpoints. 
| scanrefer | estimated| votenet | Extracts votenet estimated bounding boxes, bounding box features and global features from estimated viewpoints. 
| scanrefer | topdown | oracle | Extracts bird's eye view bounding boxes, bounding box features and global features from bird's eye viewpoints. 
---

2. Training and Evaluation
```
python main.py --train --exp_type $ET --dataset $DS --viewpoint $VP --box $BX --model $MD --visual_feat $VF...
```

where `$MD='snt'` for the Show and Tell model, and `$MD='satnt'` for Top-down and Bottom-up Attention Model. Also, `$VF` can take any combination of `'GTC'`, where it corresponds to `GLOBAL`, `TARGET` and `CONTEXT` respectively. Note that `$MD='snt'` only allows for `'GT'`. By default, `$ET` is set to `'nret'` which stands for `Non-Retrieval`. To run a retrieval experiment use `$ET='ret'`. 

other options include:
```
 --batch_size 128 
 --num_workers 16 
 --val_step 1000    
 --lr 1e-3
 --wd 1e-5
 --seed 42
```

3. Evaluation Only
```
python main.py --eval --exp_type $ET --dataset $DS --viewpoint $VP --box $BX --folder $EN
```
where `$EN` is the experiment directory name.