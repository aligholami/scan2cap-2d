# scan2cap-2d

## Setup

### Step 1 - Download & Unzip the Required Files
* Download the `hdf5` databaes for one of the following viewpoint types:

1. Annotated Viewpoint Database (Oracle Boxes):
    * [db_annotated.h5](https://mega.nz/file/eIR2CToR#795fMBvYjL9bOu4KaF5egbEn8UctsOtvrw-Rt1a5QUI)

2. Annotated Viewpoint Database (MRCNN Detected Boxes):
    * [db_annotated_mrcnn.h5](https://mega.nz/file/KcAW3J6R#r6HYeDbsa3_oWyvc3t3W4Z-xKvJ4r66i8nhYOIqLNXw

3. Estimated Viewpoint Database (3D-2D Backprojected)
    * [est.h5](https://mega.nz/file/fdYEVTwa#tvoAc2bBreaqU2i4rHeLvk7Ywzltaj6XzXTWP9wbJj0)

4. Bird's Eye Viewpoint Database (Top-Down)
    * [bev.h5](https://mega.nz/file/SIB2QTSA#z0uEWi8vZpik6O-13vSSUJoSWVzUlRtfOWI4p2C11D4)

* Download the ScanRefer `train` and `validation` splits:
    * [scanrefer.zip](https://github.com/daveredrum/ScanRefer#dataset)


and unzip the downloaded files to your desired location.
Each database contains `global features`, `object features`, `object bounding box`, `semantic label` and `object id` corresponding to each sample in the desired `ScanRefer` split. 

Alternatively, you can manually render color and instance masks and use the code provided in `preprocessing` to obtain these databases. Here is a quick guide on how to use `preprocessing` module:

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

## Training and Evaluation
Set the following paths in `scan2cap-2d/lib/conf.py` based on your needs:

```
CONF.PATH.DATA_ROOT = '/home/user/data'
CONF.PATH.CODE_ROOT = '/home/user/code'
CONF.PATH.SCANNET_DIR = "/scannet/public/v2"
```
---
Command-line arguments to run the training and/or evaluation; permutations are the same as provided in the `preprocessing` step.

    ap.add_argument("--exp_type", default="nret", help="retrieval or nonretrieval")
    ap.add_argument("--dataset", default="scanrefer", help="scanrefer or referit")
    ap.add_argument("--viewpoint", default="annotated", help="annotated, estimated or bev")
    ap.add_argument("--box", default="oracle", help="oracle, mrcnn or votenet")

1. Training and Evaluation
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

2. Evaluation Only
```
python main.py --eval --exp_type $ET --dataset $DS --viewpoint $VP --box $BX --folder $EN
```
where `$EN` is the experiment directory name.