# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness


def compute_cap_loss(data_dict, weights):
    # unpack
    pred_caps = data_dict["lang_cap"]  # (B, num_words - 1, num_vocabs)
    num_words = data_dict["lang_len"][0]
    target_caps = data_dict["lang_ids"][:, 1:num_words]  # (B, num_words - 1)
    _, _, num_vocabs = pred_caps.shape

    # caption loss
    criterion = nn.CrossEntropyLoss(ignore_index=0, weight=torch.FloatTensor(weights).cuda())
    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

    # caption acc
    pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1)  # B * (num_words - 1)
    target_caps = target_caps.reshape(-1)  # B * (num_words - 1)
    masks = target_caps != 0
    masked_pred_caps = pred_caps[masks]
    masked_target_caps = target_caps[masks]
    cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()

    return cap_loss, cap_acc


def get_scene_cap_loss(data_dict, weights):
    cap_loss, cap_acc = compute_cap_loss(data_dict, weights)

    # store
    data_dict["cap_loss"] = cap_loss
    data_dict["cap_acc"] = cap_acc

    # Final loss function
    loss = data_dict["cap_loss"]

    # Final loss function
    loss = data_dict["cap_loss"]
    data_dict["loss"] = loss

    return data_dict
