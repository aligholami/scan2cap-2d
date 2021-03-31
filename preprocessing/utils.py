import os
from random import sample
import json
import torch
import h5py
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


def validate_bbox(xyxy, width, height):
    x_min = xyxy[0]
    y_min = xyxy[1]
    x_max = xyxy[2]
    y_max = xyxy[3]
    fix = 5
    if x_max - x_min < fix:
        if x_min > fix:
            x_min -= fix
        elif x_max < width - fix:
            x_max += fix

    if y_max - y_min < fix:
        if y_min > fix:
            y_min -= fix
        elif y_max < height - fix:
            y_max += fix

    return [x_min, y_min, x_max, y_max]


def sanitize_id_coco(
        IMAGE_ID
):
    assert len(IMAGE_ID) >= 9 and len(IMAGE_ID) <= 11

    scene_id = 'scene' + IMAGE_ID[:7]
    ann_id = IMAGE_ID[-1]
    if len(IMAGE_ID) == 11:
        object_id = IMAGE_ID[7:10]

    if len(IMAGE_ID) == 10:
        object_id = IMAGE_ID[7:9]

    if len(IMAGE_ID) == 9:
        object_id = IMAGE_ID[7:8]

    sanitized = '{}-{}_{}'.format(scene_id, object_id, ann_id)
    return sanitized, scene_id, object_id, ann_id


def get_iou(box1, box2):
    """Implement the intersection over union (IoU) between box1 and box2
    
    Arguments:
    box1 -- first box, list object with coordinates (x1, y1, x2, y2)
    box2 -- second box, list object with coordinates (x1, y1, x2, y2)
    """

    # Calculate the coordinates of intersection of box1 and box2. 
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    # Calculate intersection area.
    inter_area = (y2_inter - y1_inter) * (x2_inter - x1_inter)

    # Calculate the Union area.
    box1_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    box2_area = (box1[3] - box1[1]) * (box1[2] - box1[0])
    union_area = box1_area + box2_area - inter_area

    # compute the IoU    
    iou = inter_area / union_area

    return iou


def sort_by_iou(
        SAMPLE_ID,
        SAMPLE_ID_DETECTIONS,
        GT_DB
):
    """
        Takes detections for each sample id, sorts them by IOU descending, and returns a
        SAMPLE_ID_DETECTIONS dictionary with added IOU scores.
    """
    
    try:
        sorted_detections = []
        target_object_id = int(SAMPLE_ID.split('-')[1].split('_')[0])
        gt_boxes = np.array(GT_DB['box'][SAMPLE_ID])
        gt_oids = np.array(GT_DB['objectids'][SAMPLE_ID])
        target_box_idx = np.where(gt_oids == target_object_id)[0]
        target_box = gt_boxes[target_box_idx].tolist()[0]

        # CONVERT BOX FORMAT FROM XYWH TO XYXY
        for item in SAMPLE_ID_DETECTIONS:
            detected_box = item['bbox']
            item['iou'] = get_iou(detected_box, target_box)
            sorted_detections.append(item)

        sorted_detections = sorted(sorted_detections, key=lambda x: x['iou'], reverse=True)  # descending

    except:
        print("Ignored sample {}".format(SAMPLE_ID))
        sorted_detections = None

    return sorted_detections


def export_bbox_pickle_coco(
        MRCNN_DETECTIONS_PATH,
        DB_PATH,
        GT_DB_PATH,
        RESIZE=(320, 240)
):

    pickle_dir = os.path.dirname(DB_PATH)
    os.makedirs(pickle_dir, exist_ok=True)
    db = h5py.File(DB_PATH, 'w')
    assert os.path.exists(GT_DB_PATH)
    gt_db = h5py.File(GT_DB_PATH, 'r')
    assert os.path.exists(MRCNN_DETECTIONS_PATH)
    detections = json.load(open(MRCNN_DETECTIONS_PATH))
    print("validating the mask r-cnn predictions.")
    aggregations = {}  # render_id -> list of predictions
    for pred in tqdm(detections):
        x_min, y_min, w, h = pred['bbox']
        validated_bbox = [x_min, y_min, x_min + w, y_min + h]
        validated_score = round(pred['score'], 2)
        pred['bbox'] = validated_bbox
        pred['score'] = validated_score
        sample_id, _, _, _ = sanitize_id_coco(pred['image_id'])
        if sample_id in aggregations.keys():  # filter ignored_renders
            aggregations[sample_id].append(pred)
        else:
            aggregations[sample_id] = [pred]

    # Sort based on the IoU score with the gt box in that frame/render #####
    aggregations_sorted = {}
    ditch_control = 0
    print("sorting the bounding boxes based on IoU.")
    for sample_id, detections in aggregations.items():
        res = sort_by_iou(sample_id, detections, gt_db)
        if res is not None:
            aggregations_sorted[sample_id] = res
        else:
            ditch_control += 1

    assert ditch_control <= 250

    for sample_id, detections in tqdm(aggregations_sorted.items()):
        detections = list(filter(lambda x: x['iou'] >= 0.5, detections))

        boxes = []
        ious = []
        scores = []
        categories = []
        object_ids = []
        for object_id, d in enumerate(detections):
            scale_x = RESIZE[0] // d['segmentation']['size'][0]
            scale_y = RESIZE[1] // d['segmentation']['size'][1]
            scaled_box = [scale_x * d['bbox'][0], scale_y * d['bbox'][1], scale_x * d['bbox'][2], scale_y * d['bbox'][3]]
            scaled_box = np.array(validate_bbox(scaled_box, width=RESIZE[0], height=RESIZE[1]))
            iou = np.array(d['iou'])
            score = np.array(d['score'])
            category = np.array(d['category_id'])
            object_id = np.array(object_id)


            boxes.append(scaled_box)
            ious.append(iou)
            scores.append(score)
            categories.append(category)
            object_ids.append(object_id)

        if len(boxes) >= 1:
            boxes = np.vstack(boxes)
            ious = np.vstack(ious)
            scores = np.vstack(scores)
            categories = np.vstack(categories)
            db.create_dataset('box/{}'.format(sample_id), data=boxes)
            db.create_dataset('ious/{}'.format(sample_id), data=ious)
            db.create_dataset('scores/{}'.format(sample_id), data=scores)
            db.create_dataset('objectids/{}'.format(sample_id), data=object_ids)
            db.create_dataset('categories/{}'.format(sample_id), data=categories)
    
    db.close()


def export_bbox_pickle_raw(
        AGGR_JSON_PATH,
        SCANNET_V2_TSV,
        INSTANCE_MASK_PATH,
        SAMPLE_LIST,
        SCENE_LIST,
        DB_PATH,
        RESIZE=(320, 240)
):
    id2name = get_id2name_file(AGGR_JSON=AGGR_JSON_PATH, SCENE_LIST=SCENE_LIST)
    raw2label, label2class = get_label_info(SCANNET_V2_TSV=SCANNET_V2_TSV)
    pickle_dir = os.path.dirname(DB_PATH)
    os.makedirs(pickle_dir, exist_ok=True)

    print("exporting image bounding boxes...")

    db = h5py.File(DB_PATH, 'w')
    for gg in tqdm(SAMPLE_LIST):
        sample_id = gg['sample_id']
        scene_id = gg['scene_id']

        try:
            label_img = np.array(
                Image.open(os.path.join(INSTANCE_MASK_PATH.format(scene_id, sample_id))))
            scale_x = RESIZE[0] // label_img.shape[0]
            scale_y = RESIZE[1] // label_img.shape[1]
        except FileNotFoundError as fnfe:
            print(fnfe)
            continue

        labels = np.unique(label_img)
        bbox = []
        object_ids = []
        sem_labels = []

        for label in labels:
            if label == 0: continue
            raw_name = id2name[scene_id][label - 1]
            sem_label = raw2label[raw_name]
            if raw_name in ["floor", "wall", "ceiling"]: continue
            target_coords = np.where(label_img == label)
            x_max, y_max = np.max(target_coords[1], axis=0), np.max(target_coords[0], axis=0)
            x_min, y_min = np.min(target_coords[1], axis=0), np.min(target_coords[0], axis=0)
            bbox_scaled = [x_min * scale_x, y_min * scale_y, x_max * scale_x, y_max * scale_y]
            bbox_validated = validate_bbox(bbox_scaled, RESIZE[0], RESIZE[1])
            bbox.append(np.array(bbox_validated, dtype=np.float))
            object_ids.append(np.array(label - 1, dtype=np.uint8))
            sem_labels.append(np.array(sem_label, dtype=np.uint8))

        if len(bbox) >= 1:
            bbox = np.vstack(bbox)
            oids = np.vstack(object_ids)
            slabels = np.vstack(sem_labels)
            db.create_dataset('box/{}'.format(sample_id), data=bbox)
            db.create_dataset('objectids/{}'.format(sample_id), data=oids)
            db.create_dataset('semlabels/{}'.format(sample_id), data=slabels)

    db.close()

    print("Created boxes.")


def export_image_features(
        KEY_FORMAT,
        IMAGE,
        DB_PATH,
        BOX,
        SAMPLE_LIST,
        IGNORED_SAMPLES,
        DEVICE,
        RESIZE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    fd_train = FrameData(
        ignored_samples=IGNORED_SAMPLES,
        key_format=KEY_FORMAT,
        resize=RESIZE,
        frame_path=IMAGE,
        db_path=DB_PATH,
        box=BOX,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 16,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True, device=DEVICE, mode='frame2feat').to(DEVICE)
    model.eval()

    extracted_batches = []
    print("Frame feature extraction started.")

    for i, f in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            tensor_list, bbox_list, bbox_ids, sample_id_list = f
            frame_features = model(tensor_list, None, None).to('cuda')
            extracted_batches.append(
                {
                    'frame_features': frame_features,
                    'sample_ids': sample_id_list
                }
            )

    print("Saving extracted features.")
    fd_train.write_frame_features(extracted_batches)

    return None


def export_bbox_features(
        IGNORED_SAMPLES,
        KEY_FORMAT,
        IMAGE,
        DB_PATH,
        BOX,
        SAMPLE_LIST,
        DEVICE,
        RESIZE
):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    fd_train = FrameData(
        ignored_samples=IGNORED_SAMPLES,
        key_format=KEY_FORMAT,
        resize=RESIZE,
        frame_path=IMAGE,
        db_path=DB_PATH,
        box=BOX,
        input_list=SAMPLE_LIST,
        transforms=normalize
    )

    conf = {
        'batch_size': 128,
        'num_workers': 6
    }

    data_loader = DataLoader(fd_train, collate_fn=fd_train.collate_fn, **conf)
    model = ResNet101NoFC(pretrained=True, progress=True, device=DEVICE, mode='bbox2feat').to(DEVICE)
    model.eval()

    extracted_batches = []
    print("Box feature extraction started.")

    for i, f in enumerate(tqdm(data_loader)):
        tensor_list, bbox_list, bbox_ids, sample_id_list = f

        with torch.no_grad():
            batch_feats = model(tensor_list, bbox_list, bbox_ids)
            extracted_batches.append(
                {
                    'proposals_features': batch_feats,
                    'sample_ids': sample_id_list
                }
            )

    print("Saving extracted features.")
    fd_train.write_box_features(extracted_batches)

    return None
