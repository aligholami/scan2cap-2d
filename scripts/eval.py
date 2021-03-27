import os
import argparse
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader
from lib.dataset import ScanReferDataset
from models.snt import ShowAndTell
from models.tdbu import ShowAttendAndTell
from models.retr import Retrieval2D
from lib.conf import get_config, get_samples, verify_visual_feat
from lib.eval_helper import eval_cap


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


def get_retrieval_model(args, run_config, dataset):
    _ = args
    train_scene_list = list(set([item['scene_id'] for item in dataset.sample_list]))
    scanrefer_box_features = np.load(run_config.PATH.BOX_FEAT, allow_pickle=True)
    scanrefer_train_box_features = {k: item for k, item in scanrefer_box_features.item().items() if
                                    k.split('-')[0] in train_scene_list and k.split('-')[1].split('_')[0] ==
                                    k.split('.')[1]}
    ordered_vis_feature_matrix = OrderedDict(
        [(k, v.reshape(-1, 2048)) for k, v in scanrefer_train_box_features.items()])

    model = Retrieval2D(
        vis_feat_dict=ordered_vis_feature_matrix,
        lang_ids=dataset.lang_ids
    )

    model.cuda()
    model.eval()

    return model


def eval_caption(args):
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
        shuffle=False,
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

    if args.exp_type == 'ret':
        model = get_retrieval_model(args=args, run_config=run_config, dataset=train_dset)
    elif args.exp_type == 'nret':
        model = get_model(args=args, run_config=run_config, dataset=val_dset)
    else:
        raise NotImplementedError('exp_type {} is not implemented.'.format(args.exp_type))

    # evaluate

    bleu, cider, rouge, meteor = eval_cap(
        _global_iter_id=0,
        model=model,
        dataset=val_dset,
        dataloader=val_dloader,
        phase='val',
        folder=args.folder,
        max_len=run_config.MAX_DESC_LEN,
        mode=args.exp_type,
        extras=args.extras,
        use_tf=False,
        is_eval=True
    )

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--exp_type", type=str, default='nret', help='nret or ret')
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=16)
    parser.add_argument("--num_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=2000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--criterion", type=str, default="cider",
                        help="criterion for selecting the best model [choices: bleu-1, bleu-2, bleu-3, bleu-4, cider, rouge, meteor]")
    parser.add_argument("--use_tf", action="store_true", help="enable teacher forcing in inference.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")

    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    eval_caption(args)
