import torch
import argparse
import sys
sys.path.insert(0, '../')
from preprocessing.utils import export_bbox_pickle, export_image_features, export_bbox_features
from lib.conf import get_config, get_samples

def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_type', default='nonretrieval', help='retrieval or nonretrieval')
    ap.add_argument('dataset', default='scanrefer', help='scanrefer or referit')
    ap.add_argument('viewpoint', default='annotated', help='annotated, estimated or topdown')
    ap.add_argument('box', default='oracle', help='oracle, mrcnn or votenet')

    return ap.parse_args()


def main(exp_type, dataset, viewpoint, box):
    run_config = get_config(exp_type, dataset, viewpoint, box)
    sample_list, scene_list = get_samples(mode='all', key_type=run_config.TYPES.KEY_TYPE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for the given dataset, viewpoint and box mode
    # performs the following:
    # 1. Load proper config
    # 2. Extract global ResNet101 features
    # 3. Extract bounding boxes from aggregations and instance masks
    # 4. Extract bounding box features

    # 2. Run on CPU
    export_bbox_pickle(
        AGGR_JSON_PATH=run_config.PATH.AGGR_JSON,
        SCANNET_V2_TSV=run_config.PATH.SCANNET_V2_TSV,
        INSTANCE_MASK_PATH=CONF.PATH.INSTANCE_MASK,
        SAMPLE_LIST=sample_list,
        SCENE_LIST=scene_list,
        WRITE_PICKLES_PATH=run_config.PATH.BOX
    )

    # 3. Run on Device
    export_image_features(
        IMAGE=run_config.PATH.IMAGE,
        IMAGE_FEAT=run_config.PATH.IMAGE_FEAT,
        BOX=None,
        BOX_FEAT=None,
        SAMPLE_LIST=sample_list,
        DEVICE=device
    )

    # 4. Run on Device
    export_bbox_features(
        IMAGE=run_config.PATH.IMAGE,
        IMAGE_FEAT=run_config.PATH.IMAGE_FEAT,
        BOX=run_config.PATH.BOX,
        BOX_FEAT=run_config.PATH.BOX_FEAT,
        SAMPLE_LIST=sample_list,
        DEVICE=device
    )

    # 5.


if __name__ == '__main__':
    args = parse_arg()
    main(args.exp_type, args.dataset, args.viewpoint, args.box)
