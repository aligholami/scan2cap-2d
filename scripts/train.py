import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy
from lib.dataset import ScanReferDataset
from lib.solver import Solver
from lib.config import CONF
from models.snt import ShowAndTell
from models.tdbu import ShowAttendAndTell
from lib.conf import get_config, get_samples

# Ensure reproducability
seed = 3
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def verify_visual_feat(visual_feat):
    assert ('G' in visual_feat or 'T' in visual_feat or 'C' in visual_feat)
    assert len(visual_feat) <= 3

    add_global, add_target, add_context = False, False, False
    if 'G' in visual_feat:
        add_global = True

    if 'T' in visual_feat:
        add_target = True

    if 'C' in visual_feat:
        add_context = True

    return add_global, add_target, add_context


def get_dataloader(batch_size, num_workers, shuffle, sample_list, scene_list, run_config, split):
    dataset = ScanReferDataset(
        split=split,
        sample_list=sample_list,
        scene_list=scene_list,
        run_config=run_config
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                            collate_fn=dataset.collate_fn)

    return dataset, dataloader


def get_model(args, run_config, dataset):
    model_selection = args.model
    feat_size = 0
    add_global, add_target, add_context = verify_visual_feat(args.visual_feat)

    if add_global:
        feat_size += run_config.GLOBAL_FEAT_SIZE
    if add_target:
        feat_size += run_config.TARGET_FEAT_SIZE

    assert feat_size != 0

    if add_context and model_selection == 'satnt':
        print("Using Show, Attend and Tell.")
        model = ShowAttendAndTell(
            device='cuda',
            max_desc_len=run_config.MAX_DESC_LEN,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.embedding,
            emb_size=run_config.EMBEDDING_SIZE,
            feat_size=feat_size,
            feat_input={'add_global': add_global, 'add_target': add_target},
            hidden_size=run_config.DECODER_HIDDEN_SIZE,
        )

    elif model_selection == 'snt' and not add_context:
        model = ShowAndTell(
            device='cuda',
            max_desc_len=run_config.MAX_DESC_LEN,
            training_tf=True,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.embedding,
            emb_size=run_config.EMBEDDING_SIZE,
            feat_size=feat_size,
            feat_input={'add_global': add_global, 'add_target': add_target},
            hidden_size=run_config.DECODER_HIDDEN_SIZE,
        )

    else:
        raise NotImplementedError('Requested model {} is not implemented.'.format(dataset))

    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params


def get_solver(args, run_config, dataset, dataloader):

    model = get_model(
        args=args,
        run_config=run_config,
        dataset=dataset['train']
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_" + args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    LR_DECAY_STEP = [3, 7, 10, 20, 50, 90]
    LR_DECAY_RATE = 0.6

    solver = Solver(
        args=args,
        model=model,
        config=DC,
        dataset=dataset,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        caption=not args.no_caption,
        use_tf=args.use_tf,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        criterion=args.criterion
    )
    num_params = get_num_params(model)

    return solver, num_params, root


def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["train"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def train(args):

    run_config = get_config(
        exp_type=args.exp_type,
        dataset=args.dataset,
        viewpoint=args.viewpoint,
        box=args.box
    )

    train_samples, train_scenes = get_samples(mode='train', key_type=run_config.TYPES.KEY_TYPE)
    val_samples, val_scenes = get_samples(mode='val', key_type=run_config.TYPES.KEY_TYPE)

    train_dset, train_dloader = get_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        sample_list=train_samples,
        scene_list=train_scenes,
        run_config=run_config,
        split='train'
    )

    val_dset, val_dloader = get_dataloader(

        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sample_list=val_samples,
        scene_list=val_scenes,
        run_config=run_config,
        split='val'
    )
    dataset = {
        "train": train_dset,
        "eval": {
            "val": val_dset
        }
    }
    dataloader = {
        "train": train_dloader,
        "eval": {
            "val": val_dloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, config, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--criterion", type=str, default="meteor",
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor]")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_caption", action="store_true", help="Do NOT train the caption module.")
    parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
    parser.add_argument("--training_tf", type=float, default=1)
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--include_target_bbox_feats", action="store_true")
    parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
    parser.add_argument("--only_2d", action="store_true", default=False)
    parser.add_argument("--vp_matching_train", type=str,
                        default="/local-scratch/scan2cap_extracted/vp_matching/matches_train.json")
    parser.add_argument("--vp_matching_val", type=str,
                        default="/local-scratch/scan2cap_extracted/vp_matching/matches_val.json")
    parser.add_argument('--frames_dir', nargs="?", type=str,
                        default='/project/3dlg-hcvc/scannet_extracted/sens2frame_output')
    parser.add_argument('--resnet_features_dir', nargs="?", type=str,
                        default="/project/3dlg-hcvc/gholami/vp_matching/frame_features")
    parser.add_argument('--region_features_dir', nargs="?", type=str,
                        default="/local-scratch/scan2cap_extracted/resnet101_features")
    parser.add_argument("--use_pretrained", type=str,
                        help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)
