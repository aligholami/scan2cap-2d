from lib.conf import get_config
import argparse


def parse_arg():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp_type', default='nonretrieval', help='retrieval or nonretrieval')
    ap.add_argument('dataset', default='scanrefer', help='scanrefer or referit')
    ap.add_argument('viewpoint', default='annotated', help='annotated, estimated or topdown')
    ap.add_argument('box', default='oracle', help='oracle, mrcnn or votenet')

    return ap.parse_args()


def main(exp_type, dataset, viewpoint, box):
    run_config = get_config(exp_type, dataset, viewpoint, box)

    # for the given dataset, viewpoint and box mode
    # performs the following:
    # 1. Load proper config
    # 2. Extract global ResNet101 features
    # 3. Extract bounding boxes from aggregations and instance masks
    # 4. Extract bounding box features


if __name__ == '__main__':
    args = parse_arg()
    main(args.exp_type, args.dataset, args.viewpoint, args.box)
