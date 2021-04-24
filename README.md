# scan2cap-2d

## Setup

### Step 1 - Download & Unzip the Required Files
* Download the `hdf5` databaes for one of the following viewpoint types:

1. Annotated Viewpoint Database (Oracle Boxes):
    * [db_annotated.h5](https://mega.nz/file/eIR2CToR#795fMBvYjL9bOu4KaF5egbEn8UctsOtvrw-Rt1a5QUI) (~3GB)
    
2. Annotated Viewpoint Database (MRCNN Detected Boxes):
    * [db_annotated_mrcnn.h5](https://mega.nz/file/KcAW3J6R#r6HYeDbsa3_oWyvc3t3W4Z-xKvJ4r66i8nhYOIqLNXw) (~0.5GB)

3. Estimated Viewpoint Database (3D-2D Backprojected)
    * [db_estimated.h5](https://mega.nz/file/fdYEVTwa#tvoAc2bBreaqU2i4rHeLvk7Ywzltaj6XzXTWP9wbJj0) (~2.5GB)

4. Bird's Eye Viewpoint Database (Top-Down)
    * [db_td.h5](https://mega.nz/file/SIB2QTSA#z0uEWi8vZpik6O-13vSSUJoSWVzUlRtfOWI4p2C11D4) (~11GB)

* Download the ScanRefer `train` and `validation` splits:
    * [ScanRefer Download](https://github.com/daveredrum/ScanRefer#dataset)



and unzip the downloaded files to your desired location.
Each database contains `global features`, `object features`, `object bounding box`, `semantic label` and `object id` corresponding to each sample in the desired `ScanRefer` split. 

### Optional - Prepare Databases from Scratch
Alternatively, you can manually render color and instance masks and use the code provided in `preprocessing` to obtain these databases. Make sure to set `IMAGE` and `INSTANCE_MASK` paths in the `conf.py` file. Here is a quick guide on how to use `preprocessing` module:

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

---
## Reproducing the results
Here is a set of experiments reported in the Scan2Cap paper and the commands to reproduce them. Please refer to table 6 and 8 in our paper for experiment names:
https://arxiv.org/pdf/2012.02206.pdf.

For the M2 and M2-RL results, please refer to the official [Meshed-Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer).

| Experiment           | Command | CIDER | BLEU-4 | METEOR | ROUGLE-L
|:----------------| :-----| :------| :------| :------| :------|
| {G, A, -, Retr} | ``python main.py --eval --exp_type ret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat='G' --folder exp1``| - | - | - | - | 
| {G, A, -, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle  --visual_feat 'G' --model 'snt' --folder exp2 --ckpt_path path_to_snt_checkpoint.pth`` | 51.48 | 13.47 | 20.31 | 46.81 | 
| {T, A, O, Retr} | ``python main.py --eval --exp_type ret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'T' --folder exp3`` | - | - | - | - | 
| {T+C, A, O, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'TC' --model 'satnt' --folder exp4 --ckpt_path pretrained/ant_td_tc/model.pth`` | 42.22 | 14.24 | 20.02 | 48.51 | 
| {G+T, A, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'GT' --model 'snt' --folder exp5 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 60.75 | 14.73 | 21.19 | 47.80 | 
| {G+T+C, A, O, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box oracle --visual_feat 'GTC' --model 'satnt' --folder exp6 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 15.75 | 6.76 | 16.09 | 38.98 | 
| {T+C, A, 2DM, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'TC' --model 'satnt' --folder exp7 --ckpt_path pretrained/ant_td_tc/model.pth`` | 19.41 | 8.52 | 15.93 | 38.47 | 
| {G+T, A, 2DM, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'GT' --model 'snt' --folder exp8 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 24.15 | 8.92 | 16.61 | 37.46 | 
| {G+T+C, A, 2DM, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint annotated --box mrcnn --visual_feat 'GTC' --model 'satnt' --folder exp9 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 8.00 | 4.45 | 13.59 | 31.74 | 
| {T+C, E, 3DV, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'TC' --model 'satnt' --folder exp10 --ckpt_path pretrained/ant_td_tc/model.pth`` | 31.65 | 12.10 | 18.65 | 45.31 | 
| {G+T, E, 3DV, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'GT' --model 'snt' --folder exp11 --ckpt_path pretrained/ant_snt_gt/model.pth`` | 36.59 |11.31 | 18.83| 42.54 | 
| {G+T+C, E, 3DV, TD} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint estimated --box votenet --visual_feat 'GTC' --model 'satnt' --folder exp12 --ckpt_path pretrained/ant_td_gtc/model.pth`` | 11.79 | 5.82 | 15.14 | 36.45 |
| {G, BEV, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint bev --box oracle --visual_feat 'G' --model 'snt' --folder exp13 --ckpt_path pretrained/bev_g/model.pth`` | 27.22 | 11.96 | 18.66 | 46.71 | 
| {G+T, BEV, O, S&T} | ``python main.py --eval --exp_type nret --dataset scanrefer --viewpoint bev --box oracle --visual_feat 'GT' --model 'snt' --folder exp14 --ckpt_path pretrained/bev_gt/model.pth`` | 30.31 | 13.83 | 48.07 | 19.32 | 
---