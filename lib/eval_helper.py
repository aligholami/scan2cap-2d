import os
import json
import torch
import pickle5
import numpy as np
from tqdm import tqdm
import capeval.bleu.bleu as capblue
import capeval.cider.cider as capcider
import capeval.rouge.rouge as caprouge
import capeval.meteor.meteor as capmeteor
from lib.config import CONF
from lib.loss_helper import get_scene_cap_loss

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))


def get_info_for_3d(candidates):
    # generates a list, containing the predictions on validation set with the following format.
    # [
    #     {
    #         "scene_id": "...",
    #         "object_id": "...",
    #         "ann_id": "...",
    #         "camera_pose": "...",
    #         "description": "1 caption here"
    #         "bbox_corner": ["x_min", "y_min", "x_max", "y_max"],
    #         "object_mask": "RLE FORMAT DETECTED OBJECT MASK",
    #         "depth_file_name": "..."
    #     }
    # ]
    if CONF.TYPES.IMAGE_TYPE == 'render':
        scanrefer = json.load(open(
            '/local-scratch/scan2cap_extracted/common/scanrefer/transformations/ScanRefer_filtered_fixed_viewpoint_val.json',
            'r'))

    detection_pickle = '/local-scratch/scan2cap_extracted/render-based/bbox_pickle/detected_new/boxes.p'
    detections = pickle5.load(open(detection_pickle, 'rb'))
    info_list = []
    num_failed = 0
    for candidate_key, candidate in candidates.items():
        scene_id = candidate_key.split('|')[0]
        object_id = candidate_key.split('|')[1]
        object_name = candidate_key.split('|')[2]
        ann_id = candidate_key.split('|')[3]
        camera_pose = list(filter(
            lambda x: str(x['scene_id']) == str(scene_id) and str(x['object_id']) == str(object_id) and str(
                x['ann_id']) == str(ann_id), scanrefer))
        assert len(camera_pose) == 1
        transformation = camera_pose[0]['transformation']
        description = candidate
        # print("looking for key {}-{}_{}".format(scene_id, object_id, ann_id))
        try:
            detected_bbox = [int(item) for item in
                             detections[scene_id]['{}-{}_{}'.format(scene_id, object_id, ann_id)][0]['bbox']]
            detected_mask = detections[scene_id]['{}-{}_{}'.format(scene_id, object_id, ann_id)][0]['mask']
            depth_file_name = '{}/{}-{}_{}.depth.png'.format(scene_id, scene_id, object_id, ann_id)
            info_dict = {
                'scene_id': scene_id,
                'object_id': object_id,
                'ann_id': ann_id,
                'transformation': transformation,
                'description': description,
                'detected_bbox': detected_bbox,
                'detected_mask_rle': detected_mask,
                'depth_file_name': depth_file_name
            }

            info_list.append(info_dict)
        except (IndexError, KeyError):
            num_failed += 1
            continue

    print("ignored candidates: ", num_failed)
    return info_list


def prepare_corpus(scanrefer, max_len=CONF.TRAIN.MAX_DES_LEN):
    # if CONF.TYPES.BOX_TYPE == 'votenet':
    #     corpus = {}
    #     for data in scanrefer:
    #         scene_id = data['scene_id']
    #         object_id = data['object_id']
    #         object_name = data['object_name']
    #         token = data['token'][:max_len]
    #         description = []
    #         # add start and end token
    #         for item in token:
    #             description.append("sos " + " ".join(item) + " eos")

    #         key = "{}|{}|{}".format(scene_id, object_id, object_name)
    #         corpus[key] = description

    # else:

    corpus = {}
    for data in scanrefer:
        scene_id = data['scene_id']
        object_id = data['object_id']
        object_name = data['object_name']
        token = data['token'][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus


def prepare_corpus_extra(scanrefer, max_len=CONF.TRAIN.MAX_DES_LEN):
    corpus = {}
    for data in scanrefer:
        scene_id = data['scene_id']
        object_id = data['object_id']
        ann_id = data['ann_id']
        object_name = data['object_name']
        token = data['token'][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}|{}".format(scene_id, object_id, object_name, ann_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus


def decode_caption(raw_caption, idx2word):
    if isinstance(raw_caption, type(torch.tensor([]))):
        decoded = ["sos"]
    else:
        decoded = []

    for token_idx in raw_caption:
        if isinstance(raw_caption, type(torch.tensor([]))):
            token_idx = token_idx.item()
        else:
            token_idx = int(token_idx)

        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded


def check_candidates(corpus, candidates):
    print("Number of corpus keys: ", len(corpus))
    print("Number of candidates: ", len(candidates))
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    print("Number of missing keys: ", len(missing_keys))
    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates


def check_candidate_extra(corpus_extra, candidates_extra):
    print("corps: ", len(corpus_extra))
    print("candidates_extra: ", len(candidates_extra))
    placeholder = "sos eos"
    corpus_keys = list(corpus_extra.keys())
    candidate_keys = list(candidates_extra.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    print("Number of missing keys: ", len(missing_keys))
    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates_extra[key] = [placeholder]

    return candidates_extra


def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def organize_candidates_extra(corpus_extra, candidates_extra):
    new_candidates = {}
    for key in corpus_extra.keys():
        new_candidates[key] = candidates_extra[key]

    return new_candidates


def feed_2d_oracle_cap(model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True,
                       max_len=CONF.TRAIN.MAX_DES_LEN):
    candidates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if not type(data_dict[key]) == type([]):
                data_dict[key] = data_dict[key].to('cuda')

        with torch.no_grad():
            data_dict = model(data_dict, use_tf, is_eval)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1)  # batch_size, max_len - 1
        batch_size, _ = captions.shape

        # pick out object ids of detected objects
        # detected_object_ids = data_dict["scene_object_ids"]

        # masks for the valid bboxes
        # masks = data_dict["target_masks"] # batch_size, num_proposals

        # dump generated captions
        for batch_id in range(batch_size):
            try:
                scene_id = str(data_dict['scene_id'][batch_id])
            except KeyError:

                scene_id = str(dataset.split_list[int(data_dict['scan_idx'][batch_id].item())]['scene_id'])

            target_object_id = int(data_dict['object_id'][batch_id].item())

            # ditch the -1 object ids
            if target_object_id > -1:

                target_object_name = str(dataset.id2name[scene_id][target_object_id])
                # ann_id = int(data_dict['ann_id'][batch_id])

                # for prop_id in range(num_proposals):
                caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])
                target_object_id = int(data_dict['object_id'][batch_id].item())
                key = "{}|{}|{}".format(scene_id, target_object_id, target_object_name.replace(' ', '_'))

                if key not in candidates:
                    candidates[key] = [caption_decoded]
                else:
                    candidates[key].append([caption_decoded])
            else:
                print("Skipping a key.")
                continue

    return candidates


def feed_2d_oracle_cap_new(model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True,
                           max_len=CONF.TRAIN.MAX_DES_LEN):
    candidates = {}
    candidates_extra = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            if not type(data_dict[key]) == type([]):
                data_dict[key] = data_dict[key].to('cuda')

        with torch.no_grad():
            data_dict = model(data_dict, use_tf, is_eval)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1)  # batch_size, max_len - 1
        batch_size, _ = captions.shape

        # pick out object ids of detected objects
        # detected_object_ids = data_dict["scene_object_ids"]

        # masks for the valid bboxes
        # masks = data_dict["target_masks"] # batch_size, num_proposals

        # dump generated captions
        for batch_id in range(batch_size):
            # try:
            scene_id = str(data_dict['scene_id'][batch_id])
            target_object_id = int(data_dict['object_id'][batch_id].item())
            ann_id = int(data_dict['ann_id'][batch_id].item())
            # print("ann_id: ", ann_id)
            # except KeyError as ke:
            #     print(ke)
            #     scene_id = str(dataset.split_list[int(data_dict['scan_idx'][batch_id].item())]['scene_id'])
            #     target_object_id = int(dataset.split_list[int(data_dict['scan_idx'][batch_id].item())]['object_id'])
            #     ann_id = int(dataset.split_list[int(data_dict['scan_idx'][batch_id].item())]['ann_id'])

            # ditch the -1 object ids
            # print("looking for {}-{}_{}".format(scene_id, target_object_id, ann_id))
            # print()/
            if '{}-{}_{}'.format(scene_id, target_object_id, ann_id) not in dataset.ignore_list:
                target_object_name = str(dataset.id2name[scene_id][target_object_id])
                # ann_id = int(data_dict['ann_id'][batch_id])

                # for prop_id in range(num_proposals):
                caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])
                target_object_id = int(data_dict['object_id'][batch_id].item())
                key = "{}|{}|{}".format(scene_id, target_object_id, target_object_name.replace(' ', '_'))

                # update candidates
                if key not in candidates:
                    candidates[key] = [caption_decoded]
                else:
                    candidates[key].append([caption_decoded])

                # update candidates_extras
                key_extras = "{}|{}|{}|{}".format(scene_id, target_object_id, target_object_name.replace(' ', '_'),
                                                  ann_id)
                candidates_extra[key_extras] = [caption_decoded]

            else:
                # print("DDDIDDDDDDDDDDDDDDDDDDDDDDDDDDDD!!!!!!!!!!!!!!!!!")
                print("Skipping a key.")
                continue

    return candidates, candidates_extra


def feed_2d_retreival_cap(model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True,
                          max_len=CONF.TRAIN.MAX_DES_LEN):
    assert CONF.TYPES.RETRIEVAL_MODE == True
    candidates = {}
    for data_dict in tqdm(dataloader):
        # for key in data_dict:
        #     if not type(data_dict[key]) == type([]):
        #         data_dict[key] = data_dict[key].to('cuda')

        # feed
        batch_captions = model(data_dict)

        # unpack
        captions = batch_captions
        batch_size = len(batch_captions)

        # dump generated captions
        for batch_id in range(batch_size):
            try:
                scene_id = str(data_dict['scene_id'][batch_id])
            except KeyError:

                scene_id = str(dataset.split_list[int(data_dict['scan_idx'][batch_id].item())]['scene_id'])

            target_object_id = int(data_dict['object_id'][batch_id].item())

            # ditch the -1 object ids
            if target_object_id > -1:

                target_object_name = str(dataset.id2name[scene_id][target_object_id])
                # ann_id = int(data_dict['ann_id'][batch_id])

                # for prop_id in range(num_proposals):
                caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])
                target_object_id = int(data_dict['object_id'][batch_id].item())
                key = "{}|{}|{}".format(scene_id, target_object_id, target_object_name.replace(' ', '_'))

                if key not in candidates:
                    candidates[key] = [caption_decoded]
                else:
                    candidates[key].append([caption_decoded])
            else:
                print("Skipping a key.")
                continue

    return candidates


def feed_2d_cap(model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN):
    candidates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        # feed
        data_dict = model(data_dict, use_tf, is_eval=True)
        _ = get_scene_cap_loss(data_dict, weights=dataset.weights, detection=False, caption=False)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1)  # batch_size, max_len - 1
        dataset_ids = data_dict["scan_idx"]

        batch_size, _ = captions.shape
        # dump generated captions
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            object_id = dataset.scanrefer[dataset_idx]["object_id"]
            caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])
            object_name = dataset.scanrefer[dataset_idx]["object_name"]
            key = "{}|{}|{}".format(scene_id, object_id, object_name)

            if key not in candidates:
                print("gag")
                candidates[key] = caption_decoded
            else:
                print("faf")
                candidates[key] += ' ' + caption_decoded

    return candidates


def eval_cap(_global_iter_id, model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True,
             max_len=CONF.TRAIN.MAX_DES_LEN, force=False, mode="scene"):
    # corpus
    corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
    if not os.path.exists(corpus_path) or force:
        print("preparing corpus...")
        corpus = prepare_corpus(dataset.split_list, max_len)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)

    pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
    # if not os.path.exists(pred_path) or force:
    # generate results
    with torch.no_grad():
        print("oracle top down caption mode: ...")
        # candidates = feed_2d_oracle_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # print("generating descriptions...")
        # if mode == "scene":
        #     candidates = feed_scene_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif mode == "object":
        #     candidates, cls_acc = feed_object_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif mode == '2d_no_proposal':
        #     print("Current mode: ", '2d_no_proposal')
        # candidates = feed_2d_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif args.use_oracle_bbox:
        #     print("Using oracle bboxes")
        candidates = feed_2d_oracle_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)

        # else:
        #     raise ValueError("invalid mode: {}".format(mode))
        # candidates = feed_2d_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = check_candidates(corpus, candidates)

    candidates = organize_candidates(corpus, candidates)

    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # with open(pred_path.strip('.json') + '_' + str(_global_iter_id) + '.json', "w") as f:
    #     json.dump(candidates, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save scores
    print("saving scores...")
    score_path = os.path.join(CONF.PATH.OUTPUT, folder, "score_{}.json".format(phase))
    with open(score_path, "w") as f:
        scores = {
            "bleu-1": [float(s) for s in bleu[1][0]],
            "bleu-2": [float(s) for s in bleu[1][1]],
            "bleu-3": [float(s) for s in bleu[1][2]],
            "bleu-4": [float(s) for s in bleu[1][3]],
            "cider": [float(s) for s in cider[1]],
            "rouge": [float(s) for s in rouge[1]],
            "meteor": [float(s) for s in meteor[1]],
        }
        json.dump(scores, f, indent=4)

    if mode == "scene" or mode == "2d_no_proposal":
        return bleu, cider, rouge, meteor
    else:
        return bleu, cider, rouge, meteor, np.mean(cls_acc)


def eval_cap_new(_global_iter_id, model, dataset, dataloader, phase, folder, use_tf=False, is_eval=True,
                 max_len=CONF.TRAIN.MAX_DES_LEN, force=False, mode="scene"):
    add_extras = False

    # corpus
    os.makedirs(os.path.join(CONF.PATH.EVAL_OUTPUT, folder), exist_ok=True)

    corpus_path = os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "corpus_{}.json".format(phase))

    print("preparing corpus...")
    corpus = prepare_corpus(dataset.fixed_set, max_len)
    with open(corpus_path, "w") as f:
        json.dump(corpus, f, indent=4)

    if add_extras:
        print("preparing courpus_extra...")
        corpus_extra = prepare_corpus_extra(dataset.unignored_split, max_len)
        with open(os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "corpus_extra_{}.json".format(phase)), "w") as f:
            json.dump(corpus_extra, f, indent=4)

    if add_extras:
        capper = feed_2d_oracle_cap_new
    else:
        capper = feed_2d_oracle_cap

    if CONF.TYPES.RETRIEVAL_MODE:
        capper = feed_2d_retreival_cap

    pred_path = os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "pred_{}.json".format(phase))
    # if not os.path.exists(pred_path) or force:
    # generate results
    with torch.no_grad():
        # candidates = feed_2d_oracle_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # print("generating descriptions...")
        # if mode == "scene":
        #     candidates = feed_scene_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif mode == "object":
        #     candidates, cls_acc = feed_object_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif mode == '2d_no_proposal':
        #     print("Current mode: ", '2d_no_proposal')
        # candidates = feed_2d_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # elif args.use_oracle_bbox:
        #     print("Using oracle bboxes")
        if add_extras:
            candidates, candidates_extra = capper(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        else:
            candidates = capper(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        # else:
        #     raise ValueError("invalid mode: {}".format(mode))
        # candidates = feed_2d_cap(model, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    if add_extras:
        candidates_extra = check_candidates(corpus_extra, candidates_extra)
        candidates_extra = organize_candidates(corpus_extra, candidates_extra)
        with open(os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "pred_extra_{}.json".format(phase)), "w") as f:
            json.dump(candidates_extra, f, indent=4)

        # get info list
        info_list = get_info_for_3d(candidates_extra)
        with open(os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "info_list_{}.json".format(phase)), "w") as f:
            json.dump(info_list, f, indent=4)

    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # with open(pred_path.strip('.json') + '_' + str(_global_iter_id) + '.json', "w") as f:
    #     json.dump(candidates, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save scores
    print("saving scores...")
    score_path = os.path.join(CONF.PATH.EVAL_OUTPUT, folder, "score_{}.json".format(phase))
    with open(score_path, "w") as f:
        scores = {
            "bleu-1": [float(s) for s in bleu[1][0]],
            "bleu-2": [float(s) for s in bleu[1][1]],
            "bleu-3": [float(s) for s in bleu[1][2]],
            "bleu-4": [float(s) for s in bleu[1][3]],
            "cider": [float(s) for s in cider[1]],
            "rouge": [float(s) for s in rouge[1]],
            "meteor": [float(s) for s in meteor[1]],
        }
        json.dump(scores, f, indent=4)

    if mode == "scene" or mode == "2d_no_proposal":
        return bleu, cider, rouge, meteor
    else:
        return bleu, cider, rouge, meteor, np.mean(cls_acc)