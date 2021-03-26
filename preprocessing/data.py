import os
import torch
import pickle
import math
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


class FrameData(Dataset):
    def __init__(self, resize, frame_path, frame_feature_path, box_path, box_feature_path, input_list, transforms):
        self.new_width, self.new_height = resize
        self.frames_path = frame_path
        self.frame_feature_path = frame_feature_path
        self.box_path = box_path
        self.box_feature_path = box_feature_path
        self.input_list = input_list
        self.transforms = transforms

        self.box = None
        if box_path is not None:
            with open(box_path, 'rb') as bpf:
                self.box = pickle.load(bpf)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_item = self.input_list[idx]
        sample_id = input_item['sample_id']
        scene_id = input_item['scene_id']

        frame_tensor = self.load_image(
            image_path=os.path.join(self.frames_path, scene_id, '{}.png'.format(sample_id))
        )

        # load bbox info
        if self.box is not None:
            bbox_info = self.box[sample_id]
            boxes = torch.zeros(len(bbox_info), 4).float()
            boxes_object_ids = torch.zeros(len(bbox_info)).int()
            for i, info in enumerate(bbox_info):
                scale_x = nw / ow
                scale_y = nh / oh
                xyxy_bbox_scaled = [math.floor(info["bbox"][0] * scale_x), math.floor(info["bbox"][1] * scale_y),
                                    math.ceil(info["bbox"][2] * scale_x), math.ceil(info["bbox"][3] * scale_y)]
                boxes[i] = torch.tensor(xyxy_bbox_scaled, dtype=torch.float)
                object_id = info['object_id']
                boxes_object_ids[i] = torch.tensor([object_id], dtype=torch.int16)

            ret = {
                # 'failed': False,
                'frame_tensor': frame_tensor,
                'bbox_info': boxes,
                'bbox_id': boxes_object_ids,
                'sample_id': sample_id
            }

        else:
            ret = {
                'frame_tensor': frame_tensor,
                'bbox_info': None,
                'bbox_id': None,
                'sample_id': sample_id
            }

        return ret

    def load_image(self, image_path):
        old_image = Image.open(image_path)
        resized = old_image.resize((self.new_width, self.new_height))
        rgbed = resized.convert('RGB')
        frame_tensor = torch.from_numpy(np.asarray(rgbed).astype(np.float32)).permute(2, 0, 1)

        if self.transforms:
            frame_tensor = self.transforms(frame_tensor / 255.0)

        return frame_tensor

    def collate_fn(self, data):
        # data = list(filter(lambda x: x['use'], data))
        tensor_list = [d['frame_tensor'] for d in data]
        bbox_list = [d['bbox_info'] for d in data]
        bbox_ids = [d['bbox_id'] for d in data]
        sample_id_list = [d['sample_id'] for d in data]

        return tensor_list, bbox_list, bbox_ids, sample_id_list

    def write_frame_features(self, batches):
        """
            writes extracted features as numpy arrays.
            batches is a list of dict. Each dict has a batch of results
            in it.
        """
        target_npy = self.frame_feature_path
        aggregation = {}
        target_dir = os.path.dirname(target_npy)
        os.makedirs(target_dir, exist_ok=True)

        for batch in tqdm(batches):
            batch_size = len(batch['sample_ids'])
            for i in range(batch_size):
                k = '{}'.format(batch['sample_ids'][i])
                aggregation[k] = batch['frame_features'][i]

        np.save(target_npy, aggregation)

    def write_box_features(self, batches):
        """
            writes extracted features as numpy arrays.
            batches is a list of dict. Each dict has a batch of results
            in it.
        """
        target_npy = self.box_feature_path
        aggregation = {}
        target_dir = os.path.dirname(target_npy)
        os.makedirs(target_dir, exist_ok=True)

        for batch in tqdm(batches):
            batch_size = len(batch['sample_ids'])
            for i in range(batch_size):
                frame_object_features = batch['proposals_features'][i]
                sample_id = batch['sample_ids'][i]
                sample_box_features = {}
                for object_id, feature in frame_object_features.items():
                    sample_box_features[object_id] = feature.squeeze().cpu().numpy()
                aggregation[sample_id] = sample_box_features
        np.save(target_npy, aggregation)
