import os
import random
import time
import json
import math
import torch
import imagesize
import numpy as np
from tqdm import tqdm
import pickle5 as pickle
from itertools import chain
from collections import Counter
from torch.utils.data import Dataset
import torch.utils.data as data_tools

class ScanReferDataset(Dataset):

    def __init__(self,
                 split,
                 sample_list,
                 scene_list,
                 run_config
                 ):

        self.split = split
        self.sample_list = sample_list
        self.scene_list = scene_list
        self.run_config = run_config
        self.vocabulary = None
        self.weights = None
        self.glove = None
        self.lang = None
        self.lang_ids = None
        self.id2name = self.get_id2name_file()
        self.load_data()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        start = time.time()
        item = self.sample_list[idx]
        sample_id = item['sample_id']
        lang_feat = self.lang[sample_id]
        lang_ids = np.array(self.lang_ids[sample_id])
        lang_len = len(item["token"]) + 2
        lang_len = lang_len if lang_len <= self.run_config.MAX_DESC_LEN + 2 else self.run_config.MAX_DESC_LEN + 2

        with open(self.run_config.PATH.BOX, 'rb') as f:
            box = pickle.load(f)
            box = box[sample_id]

        with open(self.run_config.PATH.BOX_FEAT, 'rb') as f:
            box_feat = np.load(f)
            box_feat = box_feat[sample_id]

        with open(self.run_config.PATH.FRAME_FEAT, 'rb') as f:
            frame_feat = np.load(f)

        pool_ids = []
        pool_feats = []
        for ix, bbox_info in enumerate(box):
            object_id = bbox_info['object_id']  # true if bbox_type is oracle, otherwise it only has the target object id
            xyxy_bbox = np.array(
                [math.floor(bbox_info["bbox"][0]), math.floor(bbox_info["bbox"][1]),
                 math.ceil(bbox_info["bbox"][2]), math.ceil(bbox_info["bbox"][3])])

            target_object_feat = np.concatenate((target_object_feat, xyxy_bbox))

            pool_ids.append(object_id)
            pool_feats.append(box_feat)

        ret = {
            'use': True,
            'lang_feat': None,
            'lang_len': None,
            'lang_ids': None,
            'vis_feats': None,
            'sample_ids': None,
            "target_object_feat": None,
            "proposals_object_ids_list": None,
            "proposals_feats_list": None,
            'load_time': time.time() - start
        }

        return ret

    def collate_fn(self):
        pass

    def load_data(self):
        pass

    def get_raw2label(self):
        # mapping
        scannet_labels = DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
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

        return raw2label

    def get_id2name_file(self):
        print("getting id2name...")
        id2name = {}
        item_ids = []
        for scene_id in tqdm(self.scene_list):
            id2name[scene_id] = {}
            aggr_file = json.load(open(self.run_config.PATH.AGGR_JSON.format(scene_id, scene_id)))
            for item in aggr_file["segGroups"]:
                item_ids.append(int(item["id"]))
                id2name[scene_id][int(item["id"])] = item["label"]

        return id2name

    def get_label_info(self):
        label2class = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                       'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                       'refridgerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}

        # mapping
        scannet_labels = label2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}
        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
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

    def transform_des(self):
        lang = {}
        label = {}
        max_len = self.run_config.MAX_DESC_LEN
        for data in self.sample_list:
            sample_id = data['sample_id']

            if sample_id not in lang:
                lang[sample_id] = {}
                label[sample_id] = {}

            # trim long descriptions
            tokens = data["token"][:max_len]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((max_len + 2, 300))
            labels = np.zeros((max_len + 2))  # start and end

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]

            # store
            lang[sample_id] = embeddings
            label[sample_id] = labels

        return lang, label

    def build_vocab(self):
        if os.path.exists(self.run_config.PATH.SCANREFER_VOCAB):
            self.vocabulary = json.load(open(self.run_config.PATH.SCANREFER_VOCAB))
            print("Loaded the existing vocabulary.")
            print("Number of keys in the vocab: {}".format(len(self.vocabulary['idx2word'].keys())))

        else:
            if self.split == "train":
                all_words = chain(*[data["token"][:self.run_config.MAX_DESC_LEN] for data in self.sample_list])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove],
                                      key=lambda x: x[1],
                                      reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos",
                       "eos"]  # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(self.run_config.PATH.SCANREFER_VOCAB, "w"), indent=4)

                self.vocabulary = vocab

    def build_frequency(self):
        if os.path.exists(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS):
            with open(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            weights = np.ones((len(word_count)))
            self.weights = weights

            with open(self.run_config.PATH.SCANREFER_VOCAB_WEIGHTS, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)

    def _load_data(self):
        print("loading data...")
        # load language features
        self.glove = pickle.load(open(self.run_config.PATH.GLOVE_PICKLE, "rb"))
        self.build_vocab()
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self.transform_des()
        self.build_frequency()

        # add scannet data
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.sample_list])))

        # prepare class mapping
        lines = [line.rstrip() for line in open(self.run_config.PATH.SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self.get_raw2label()

    def collate_fn(self, data):

        data = filter(lambda d: d['failed'] == False, data)
        data_dicts = sorted(data, key=lambda d: len(d['proposals_object_ids_list']), reverse=True)
        max_proposals_in_batch = len(data_dicts[0]['proposals_object_ids_list'])
        proposal_feature_shape = 2052
        global_feature_size = 2048
        batch_size = len(data_dicts)
        lang_feat = torch.zeros((batch_size, len(data_dicts[0]['lang_feat']), len(data_dicts[0]['lang_feat'][0])),
                                dtype=torch.float32)
        lang_len = torch.zeros((batch_size, 1), dtype=torch.int16)
        lang_ids = torch.zeros((batch_size, len(data_dicts[0]['lang_ids'])), dtype=torch.long)
        scene_id = []
        ann_id = torch.zeros((batch_size, 1), dtype=torch.int16)
        object_id = torch.zeros((batch_size, 1), dtype=torch.int16)
        vis_feats = torch.zeros((batch_size, 2048), dtype=torch.float)
        target_object_feat = torch.zeros((batch_size, proposal_feature_shape))
        padded_proposal_feat = torch.zeros((batch_size, max_proposals_in_batch, proposal_feature_shape))
        padded_proposal_object_ids = torch.zeros((batch_size, max_proposals_in_batch, 1), dtype=torch.int16)
        padded_proposal_object_ids[:, :, :] = -1
        times = torch.zeros((batch_size, 1))

        for ix, d in enumerate(data_dicts):
            num_proposals = len(d['proposals_object_ids_list'])
            padded_proposal_feat[ix, :num_proposals, :] = torch.from_numpy(
                np.vstack(d['proposals_feats_list'])).unsqueeze(0)
            padded_proposal_object_ids[ix, :num_proposals, :] = torch.from_numpy(
                np.vstack(d['proposals_object_ids_list']))
            target_object_feat[ix, :] = torch.from_numpy(d['target_object_feat']).unsqueeze(0)
            vis_feats[ix, :] = torch.from_numpy(d['vis_feats']).squeeze().unsqueeze(0)
            lang_feat[ix, :] = torch.tensor(d['lang_feat'])
            lang_len[ix, :] = torch.tensor(d['lang_len'])
            lang_ids[ix, :] = torch.tensor(d['lang_ids'])
            scene_id.append(d['scene_id'])
            ann_id[ix, :] = torch.tensor(d['ann_id'])
            object_id[ix, :] = torch.tensor(d['object_id'])
            times[ix, :] = d['load_time']

        return {
            "lang_feat": lang_feat,
            "lang_len": lang_len,
            "lang_ids": lang_ids,
            "scene_id": scene_id,
            "ann_id": ann_id,
            "object_id": object_id,
            "vis_feats": vis_feats,
            "target_object_feat": target_object_feat,
            "proposals_object_ids": padded_proposal_object_ids,
            "proposals_feats": padded_proposal_feat,
            "load_time": times
        }
