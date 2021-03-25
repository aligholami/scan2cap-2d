import os
import json
import torch
from tqdm import tqdm
import capeval.bleu.bleu as capblue
import capeval.cider.cider as capcider
import capeval.rouge.rouge as caprouge
import capeval.meteor.meteor as capmeteor

# def get_info_for_3d(candidates):
#     """
#         generates a list, containing the predictions on validation set with the following format.
#         [
#             {
#                 "scene_id": "...",
#                 "object_id": "...",
#                 "ann_id": "...",
#                 "camera_pose": "...",
#                 "description": "1 caption here"
#                 "bbox_corner": ["x_min", "y_min", "x_max", "y_max"],
#                 "object_mask": "RLE FORMAT DETECTED OBJECT MASK",
#                 "depth_file_name": "..."
#             }
#         ]
#     :param candidates: dictionary mapping from keys to captions.
#     :return: info_list, described above.
#     """
#
#     if CONF.TYPES.IMAGE_TYPE == 'render':
#         scanrefer = json.load(open(
#             '/local-scratch/scan2cap_extracted/common/scanrefer/transformations/ScanRefer_filtered_fixed_viewpoint_val.json',
#             'r'))
#
#     detection_pickle = '/local-scratch/scan2cap_extracted/render-based/bbox_pickle/detected_new/boxes.p'
#     detections = pickle.load(open(detection_pickle, 'rb'))
#     info_list = []
#     num_failed = 0
#     for sample_id, captions in candidates.items():
#
#         camera_pose = list(filter(
#             lambda x: str(x['scene_id']) == str(scene_id) and str(x['object_id']) == str(object_id) and str(
#                 x['ann_id']) == str(ann_id), scanrefer))
#         assert len(camera_pose) == 1
#         transformation = camera_pose[0]['transformation']
#         description = candidate
#
#         try:
#             detected_bbox = [int(item) for item in
#                              detections[scene_id]['{}-{}_{}'.format(scene_id, object_id, ann_id)][0]['bbox']]
#             detected_mask = detections[scene_id]['{}-{}_{}'.format(scene_id, object_id, ann_id)][0]['mask']
#             depth_file_name = '{}/{}-{}_{}.depth.png'.format(scene_id, scene_id, object_id, ann_id)
#             info_dict = {
#                 'scene_id': scene_id,
#                 'object_id': object_id,
#                 'ann_id': ann_id,
#                 'transformation': transformation,
#                 'description': description,
#                 'detected_bbox': detected_bbox,
#                 'detected_mask_rle': detected_mask,
#                 'depth_file_name': depth_file_name
#             }
#
#             info_list.append(info_dict)
#         except (IndexError, KeyError):
#             num_failed += 1
#             continue
#
#     print("ignored candidates: ", num_failed)
#     return info_list
#

def prepare_corpus(scanrefer, max_len):
    corpus = {}
    for data in scanrefer:
        sample_id = data['sample_id']
        token = data['token'][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        if sample_id not in corpus:
            corpus[sample_id] = []

        corpus[sample_id].append(description)

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
        if token == "eos":
            break

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


def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates


def update_candidates(captions, candidates, data_dict, dataset):
    batch_size, _ = captions.shape
    # dump generated captions
    for batch_id in range(batch_size):
        sample_id = data_dict['sample_id'][batch_id]
        caption_decoded = decode_caption(captions[batch_id], dataset.vocabulary["idx2word"])
        if sample_id not in candidates:
            candidates[sample_id] = [caption_decoded]
        else:
            candidates[sample_id].append([caption_decoded])

    return candidates


def feed_2d_cap(model, dataset, dataloader, use_tf=False, is_eval=True):
    candidates = {}
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            if not type(data_dict[key]) == type([]):
                data_dict[key] = data_dict[key].to('cuda')

        data_dict = model(data_dict, use_tf, is_eval)
        captions = data_dict["lang_cap"].argmax(-1)  # batch_size, max_len - 1
        candidates = update_candidates(captions, candidates, data_dict, dataset)

    return candidates


def feed_2d_retrieval_cap(model, dataset, dataloader):
    candidates = {}
    for data_dict in tqdm(dataloader):
        captions = model(data_dict)
        candidates = update_candidates(captions, candidates, data_dict, dataset)

    return candidates


def eval_cap(_global_iter_id,
             model,
             dataset,
             dataloader,
             phase,
             folder,
             max_len,
             mode,
             extras=False,
             use_tf=False,
             is_eval=True
             ):

    # corpus
    run_config = dataset.run_config
    corpus_path = os.path.join(run_config.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
    if not os.path.exists(corpus_path):
        print("preparing corpus...")
        corpus = prepare_corpus(dataset.split_list, max_len)
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)
    else:
        print("loading corpus...")
        with open(corpus_path) as f:
            corpus = json.load(f)

    with torch.no_grad():
        if mode == 'nonretrieval':
            candidates = feed_2d_cap(model, dataset, dataloader, use_tf, is_eval)
        elif mode == 'retrieval':
            candidates = feed_2d_retrieval_cap(model, dataset, dataloader)

    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    pred_path = os.path.join(run_config.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    if extras:
        extras = dataset.get_candidate_extras(candidates)
        with open(os.path.join(run_config.PATH.OUTPUT, folder, "extras_{}.json".format(phase)), "w") as f:
            json.dump(extras, f, indent=4)

    # compute scores
    print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save scores
    print("saving scores...")
    score_path = os.path.join(run_config.PATH.OUTPUT, folder, "score_{}.json".format(phase))
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

    return bleu, cider, rouge, meteor
