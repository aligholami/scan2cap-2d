import os
import time
import json
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from preprocessing.data import FrameData
from preprocessing.model import ResNet101NoFC


def get_id2name_file(AGGR_JSON, SCENE_LIST):
    print("getting id2name...")
    id2name = {}
    item_ids = []
    all_scenes = SCENE_LIST
    print("Number of scenes: ", len(all_scenes))
    for scene_id in tqdm(all_scenes):
        id2name[scene_id] = {}
        aggr_file = json.load(open(AGGR_JSON.format(scene_id, scene_id)))
        for item in aggr_file["segGroups"]:
            item_ids.append(int(item["id"]))
            id2name[scene_id][int(item["id"])] = item["label"]

    return id2name


def get_label_info(SCANNET_V2_TSV):
    label2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                   'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                   'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}

    # mapping
    scannet_labels = label2class.keys()
    scannet2label = {label: i for i, label in enumerate(scannet_labels)}

    lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
    lines = lines[1:]
    raw2label = {}
    for i in range(len(lines)):
        label_classes_set = set(scannet_labels)
        elements = lines[i].split('\t')
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2label[raw_name] = scannet2label['others']
        else:
            raw2label[raw_name] = scannet2label[nyu40_name]

    return raw2label, label2class


def export_bbox_pickle(
        AGGR_JSON_PATH,
        SCANNET_V2_TSV,
        INSTANCE_MASK_PATH,
        SAMPLE_LIST,
        SCENE_LIST,
        WRITE_PICKLES_PATH
):
    id2name = get_id2name_file(AGGR_JSON=AGGR_JSON_PATH, SCENE_LIST=SCENE_LIST)
    raw2label, label2class = get_label_info(SCANNET_V2_TSV=SCANNET_V2_TSV)
    pickle_dir = os.path.dirname(WRITE_PICKLES_PATH)
    scattered_pickles_dir = os.path.join(pickle_dir, 'temp')
    os.makedirs(scattered_pickles_dir, exist_ok=True)

    print("exporting image bounding boxes...")

    aggregation = {}
    for gg in tqdm(SAMPLE_LIST):
        scene_id = gg['scene_id']
        object_id = gg['object_id']
        ann_id = gg['ann_id']
        try:
            label_img = np.array(Image.open(os.path.join(INSTANCE_MASK_PATH, scene_id, '{}-{}_{}.png'.format(scene_id, object_id, ann_id))))
        except FileNotFoundError as fnfe:
            print(fnfe)
            continue

        labels = np.unique(label_img)
        bbox_info = []
        for label in labels:
            if label == 0: continue
            raw_name = id2name[scene_id][label - 1]
            sem_label = raw2label[raw_name]
            if raw_name in ["floor", "wall", "ceiling"]: continue
            target_coords = np.where(label_img == label)
            x_max, y_max = np.max(target_coords[1], axis=0), np.max(target_coords[0], axis=0)
            x_min, y_min = np.min(target_coords[1], axis=0), np.min(target_coords[0], axis=0)

            bbox_info.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "object_id": label - 1,
                    "object_name": raw_name,
                    "sem_label": sem_label
                }
            )

            sample_bbox_info = {
                '{}-{}_{}'.format(scene_id, object_id, ann_id): bbox_info
            }
            try:
                aggregation[scene_id].append(sample_bbox_info)
            except KeyError:
                aggregation[scene_id] = [sample_bbox_info]

    with open(WRITE_PICKLES_PATH, "wb") as f:
        pickle.dump(aggregation, f)

    print("Created boxes.")


def export_image_features(
        IMAGE,
        IMAGE_FEAT,
        BOX,
        BOX_FEAT,
        SAMPLE_LIST,
        DEVICE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    fd_train = FrameData(
        frame_path=IMAGE,
        frame_feature_path=IMAGE_FEAT,
        box_path=BOX,
        box_feature_path=BOX_FEAT,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 16,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True).to(DEVICE)
    model.eval()

    extracted_batches = []
    print("Frame feature extraction started.")

    for i, f in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            tensor_list, bbox_list, bbox_ids, scene_list, object_list, ann_list = f
            frame_features = model(tensor_list).to('cuda')
            extracted_batches.append(
                {
                    'frame_features': frame_features.detach().cpu(),
                    'scene_id': scene_list,
                    'object_id': object_list,
                    'ann_id': ann_list
                }
            )

    print("Saving extracted features.")
    fd_train.write_frame_features(extracted_batches)

    return None


def export_bbox_features(
        IMAGE,
        IMAGE_FEAT,
        BOX,
        BOX_FEAT,
        SAMPLE_LIST,
        DEVICE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    fd_train = FrameData(
        frame_path=IMAGE,
        frame_feature_path=IMAGE_FEAT,
        box_path=BOX,
        box_feature_path=BOX_FEAT,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 128,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True).to(DEVICE)
    model.eval()

    extracted_batches = []
    print("Frame feature extraction started.")

    for i, f in enumerate(tqdm(data_loader)):
        tensor_list, bbox_list, bbox_ids, scene_list, object_list, ann_list = f

        with torch.no_grad():
            batch_feats = model(tensor_list, bbox_list, bbox_ids)
            extracted_batches.append(
                {
                    'proposals_features': batch_feats,
                    'scene_id': scene_list,
                    'object_id': object_list,
                    'ann_id': ann_list
                }
            )

    print("Saving extracted features.")
    fd_train.write_box_features(extracted_batches)

    return None
