import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch
from scipy.spatial import distance
import numpy as np
from tempfile import mkdtemp
from numpy import tensordot
from numpy.linalg import norm
import os.path as PH

class Retrieval2D(nn.Module):
    def __init__(self, vis_feat_dict, lang_ids):
        super().__init__()
        self.vis_feat_dict = vis_feat_dict
        self.N_REPEATS = len(list(self.vis_feat_dict.keys()))
        fname_train = PH.join(mkdtemp(), 'tempfile_train.dat')
        self.memmap_obj_train = np.memmap(fname_train, dtype="float32", mode="w+", shape=(self.N_REPEATS, 2048))
        for i, tns in enumerate(self.vis_feat_dict.values()):
            self.memmap_obj_train[i, :] = tns

        self.memmap_obj_train = np.memmap(fname_train, dtype="float32", mode="r", shape=(self.N_REPEATS, 2048))
        # self.fname_val = PH.join(mkdtemp(), 'tempfile_val.dat')
        self.lang_ids = lang_ids

    def get_best_rank_id(self, vis_feat):
        ranks = {}
        # max_cosine_val = -100.0
        # max_cosine_key = None
        # vis_feat = vis_feat.view(1, 2048)
        # memmap_obj_val_repeated = np.memmap(self.fname_val, dtype="float32", mode="w+", shape=(self.N_REPEATS, 2048))
        # for i, tns in enumerate(range(vis_feat.shape[0])):
        #     memmap_obj_val_repeated[i, :] = vis_feat

        # memmap_obj_val_repeated = np.memmap(self.fname_val, dtype="float32", mode="r", shape=(self.N_REPEATS, 2048))
        memmap_obj_val_repeated = vis_feat.repeat(self.N_REPEATS, 1)
        a = memmap_obj_val_repeated
        b = self.memmap_obj_train

        # cosine_ranked = tensordot(a, b)/(norm(a, axis=1)*norm(b, axis=1))
        cosine_ranked = np.einsum('xy,xy->x', a, b) / (norm(a, axis=1)*norm(b, axis=1))
        max_cosine_arg = np.argmax(cosine_ranked)
        # print("arg: ", max_cosine_arg)
        k = list(self.vis_feat_dict.keys())[max_cosine_arg]

        scene_id = k.split('-')[0]
        target_object_id = k.split('-')[1].split('_')[0]
        ann_id = k.split('-')[1].split('_')[1].split('.')[0]

        return scene_id, target_object_id, ann_id

    def forward(self, data_dict):
        vis_feats = data_dict['target_object_feat'] # batch_size, feat_size
        # print("vis ", vis_feats.shape)
        batch_size = vis_feats.shape[0]

        batch_captions = []
        for i in range(batch_size):
            # perform a global cosine ranking
            vis_feat = vis_feats[i, :]
            r_scene_id, r_object_id, r_ann_id = self.get_best_rank_id(vis_feat)
            caption = self.lang_ids[r_scene_id][r_object_id][r_ann_id]
            batch_captions.append(caption)

        return batch_captions # batch_size, num_words
