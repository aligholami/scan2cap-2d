import torch
import numpy as np
from torch.utils.data import Dataset


class FrameData(Dataset):
    def __init__(self, frame_path, frame_feature_path, box_path, box_feature_path, input_list, transforms, split):
        self.frames_path = frame_path
        self.frame_feature_path = frame_feature_path
        self.box_path = box_path
        self.box_feature_path = box_feature_path
        self.split = split
        self.scene_list_dir = args.scene_list_dir
        self.input_list = input_list
        self.transforms = transforms

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_item = self.input_list[idx]
        scene_id = input_item['scene_id']
        target_object_id = input_item['object_id']
        ann_id = input_item['ann_id']

        frame_tensor = self.load_image(
            image_path=self.frames_path.format(f['scene_id'], f['scene_id'], str(f['object_id']), str(f['ann_id'])))

        # load bbox info
        with open(BBOX_PATH.format(scene_id, scene_id, target_object_id, ann_id), "rb") as f:
            bbox_info = pickle.load(f)
            boxes = torch.zeros(len(bbox_info), 4).float()
            boxes_object_ids = torch.zeros(len(bbox_info)).int()
            for i, info in enumerate(bbox_info):
                scale_x = new_width / old_width
                scale_y = new_height / old_height
                xyxy_bbox_scaled = [math.floor(info["bbox"][0] * scale_x), math.floor(info["bbox"][1] * scale_y),
                                    math.ceil(info["bbox"][2] * scale_x), math.ceil(info["bbox"][3] * scale_y)]
                xyxy_bbox_validated = self.validate_bbox(xyxy_bbox_scaled, new_width, new_height)
                boxes[i] = torch.FloatTensor(xyxy_bbox_validated)
                object_id = info['object_id']

                boxes_object_ids[i] = torch.IntTensor([object_id])

        ret = {
            'failed': False,
            'image_tensor': frame_tensor,
            'bbox_info': boxes,
            'bbox_id': boxes_object_ids,
            'scene_id': scene_id,
            'object_id': target_object_id,
            'ann_id': ann_id
        }

        return ret

    def validate_bbox(self, xyxy, width, height):
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

        xyxy_updated = [x_min, y_min, x_max, y_max]
        return xyxy_updated

    def load_image(self, image_path):

        resized = Image.open(image_path).resize((224, 224))
        rgbed = resized.convert('RGB')
        frame_tensor = torch.from_numpy(np.asarray(rgbed).astype(np.float32)).permute(2, 0, 1)

        if self.transforms:
            frame_tensor = self.transforms(frame_tensor / 255.0)

        return frame_tensor

    def collate_frame(self, data):
        data = list(filter(lambda x: x['use'] == True, data))
        return data_tools.dataloader.default_collate(data)

    def collate_frame_box(self, data):

        tensor_list = [d['frame_tensor'] for d in data]
        bbox_list = [d['bbox_info'] for d in data]
        bbox_ids = [d['bbox_id'] for d in data]
        scene_list = [d['scene_id'] for d in data]
        object_list = [d['object_id'] for d in data]
        ann_list = [d['ann_id'] for d in data]

        return tensor_list, bbox_list, bbox_ids, scene_list, object_list, ann_list

    def write_frame_features(self, batches):
        """
            writes extracted features as numpy arrays.
            batches is a list of dict. Each dict has a batch of results
            in it.
        """

        for batch in tqdm(batches):
            batch_size = len(batch['scene_id'])
            for i in range(batch_size):
                write_dir = self.write_features_path.format(str(batch['scene_id'][i]))
                if not os.path.isdir(write_dir):
                    os.makedirs(write_dir)
                lpath = os.path.join(write_dir,
                                     str(batch['scene_id'][i]) + '-' + batch['object_id'][i] + '_' + batch['ann_id'][
                                         i] + '.npy')
                np.save(lpath, batch['frame_features'][i])

    def write_box_features(self, batches):
        """
            writes extracted features as numpy arrays.
            batches is a list of dict. Each dict has a batch of results
            in it.
        """
        for batch in tqdm(batches):
            batch_size = len(batch['scene_id'])
            # for each frame in batch
            for i in range(batch_size):
                # for each bbox in frame
                frame_object_features = batch['proposals_features'][i]
                scene_id = batch['scene_id'][i]
                target_object_id = batch['object_id'][i]
                ann_id = batch['ann_id'][i]
                for object_id, feature in frame_object_features.items():
                    # save in the following format (frame_idx = scene_id-target_object_id_annid).object_id
                    write_dir = self.args.features_dir.format(str(batch['scene_id'][i]))
                    if not os.path.isdir(write_dir):
                        os.makedirs(write_dir)
                    dump_name = '{}-{}_{}.{}.npy'.format(scene_id, target_object_id, ann_id, object_id)
                    lpath = os.path.join(write_dir, dump_name)
                    np.save(lpath, feature.squeeze().cpu().numpy())
