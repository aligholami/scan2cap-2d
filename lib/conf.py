import os
import json
from easydict import EasyDict

#########################
CONF = EasyDict()
CONF.SCAN_WIDTH = 320
CONF.SCAN_HEIGHT = 240
CONF.MAX_DESC_LEN = 30
CONF.EMBEDDING_SIZE = 300
CONF.DECODER_HIDDEN_SIZE = 512
CONF.GLOBAL_FEATURE_SIZE = 2048
CONF.TARGET_FEATURE_SIZE = 2052
CONF.PROPOSAL_FEATURE_SIZE = 2052
CONF.NUM_PROPOSALS = None
CONF.LABEL2CLASS = {'cabinet': 0, 'bed': 1, 'chair': 2, 'sofa': 3, 'table': 4, 'door': 5,
                    'window': 6, 'bookshelf': 7, 'picture': 8, 'counter': 9, 'desk': 10, 'curtain': 11,
                    'refrigerator': 12, 'shower curtain': 13, 'toilet': 14, 'sink': 15, 'bathtub': 16, 'others': 17}

CONF.PATH = EasyDict()
CONF.PATH.DATA_ROOT = '/local-scratch/code/scan2cap_extracted'
CONF.PATH.CODE_ROOT = '/local-scratch/projects/scan2cap-2d'
CONF.PATH.OUTPUT_ROOT = os.path.join(CONF.PATH.CODE_ROOT, "outputs")
CONF.PATH.SCANNET_DIR = "/datasets/released/scannet/public/v2"
CONF.PATH.SCANS_DIR = os.path.join(CONF.PATH.SCANNET_DIR, "scans")
CONF.PATH.AGGR_JSON = os.path.join(CONF.PATH.SCANS_DIR, "{}/{}_vh_clean.aggregation.json")  # scene_id, scene_id
CONF.PATH.SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_DIR, 'scannet-labels.combined.tsv')
CONF.PATH.SCANREFER_TRAIN = os.path.join(CONF.PATH.DATA_ROOT, 'ScanRefer_filtered_fixed_viewpoint_train.json')
CONF.PATH.SCANREFER_VAL = os.path.join(CONF.PATH.DATA_ROOT, 'ScanRefer_filtered_fixed_viewpoint_val.json')
CONF.PATH.SCANREFER_VOCAB = os.path.join(CONF.PATH.DATA_ROOT, 'ScanRefer_vocabulary.json')
CONF.PATH.SCANREFER_VOCAB_WEIGHTS = os.path.join(CONF.PATH.DATA_ROOT, 'ScanRefer_vocabulary_weights.json')
CONF.PATH.GLOVE_PICKLE = os.path.join(CONF.PATH.DATA_ROOT, 'glove.p')


#########################

def adapt_sample_keys(sample_list, key_type):
    """
        Converts sample_list to a new format where instead of "scene_id", "object_id" and "ann_id"
        there is a "sample_id".
    :param key_type:
    'kkk' for {scene_id}-{object_id}_{ann_id}
    'kk' for {scene_id}-{object_id}
    'k' for {scene_id}
    :return: new sample list.
    """
    assert key_type in ['kkk', 'kk', 'k']

    up_sl = []
    for item in sample_list:
        if key_type == 'kkk':
            key_format = '{}-{}_{}'
            item['sample_id'] = key_format.format(item['scene_id'], item['object_id'], item['ann_id'])
            up_sl.append(item)

        elif key_type == 'kk':
            key_format = '{}-{}'
            item['sample_id'] = key_format.format(item['scene_id'], item['object_id'])
            up_sl.append(item)

        elif key_type == 'k':
            key_format = '{}'
            item['sample_id'] = key_format.format(item['scene_id'])
            up_sl.append(item)

        else:
            pass

    return up_sl


def get_samples(mode, key_type):
    subset = True
    if subset:
        subset_range = 1000
    else:
        subset_range = None

    assert mode in ['train', 'val', 'all']
    sample_list = []
    scene_list = []
    if mode == 'train':
        t = json.load(open(CONF.PATH.SCANREFER_TRAIN))[:subset_range]
        sample_list = t 
        scene_list = list(set([item['scene_id'] for item in sample_list]))

    if mode == 'val':
        v = json.load(open(CONF.PATH.SCANREFER_VAL))[:subset_range]
        sample_list = v
        scene_list = list(set([item['scene_id'] for item in sample_list]))

    if mode == 'all':
        t = json.load(open(CONF.PATH.SCANREFER_TRAIN))[:subset_range]
        v = json.load(open(CONF.PATH.SCANREFER_VAL))[:subset_range]
        sample_list = t + v
        scene_list = list(set([item['scene_id'] for item in sample_list]))

    sample_list = adapt_sample_keys(sample_list, key_type)
    return sample_list, scene_list


def get_config(exp_type, dataset, viewpoint, box):
    CONF.MODES = EasyDict()
    CONF.TYPES = EasyDict()
    assert exp_type in ['nret', 'ret']
    assert dataset in ['scanrefer', 'referit']
    assert viewpoint in ['annotated', 'estimated', 'topdown']
    assert box in ['oracle', 'mrcnn', 'votenet']

    selected_dataset = dataset
    selected_viewpoint = viewpoint
    selected_box_mode = box
    data_root = CONF.PATH.DATA_ROOT
    code_root = CONF.PATH.CODE_ROOT
    output_root = CONF.PATH.OUTPUT_ROOT

    if selected_dataset == 'scanrefer':
        if selected_viewpoint == 'annotated':
            CONF.PATH.IMAGE = os.path.join(data_root, 'render-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'tatt-based/resnet101_features.npy')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'render-based/instance-masks/{}/{}.objectId.encoded.png')
            CONF.PATH.BOX = os.path.join(data_root, 'tatt-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'tatt-based/box_feat/box_feat.npy')
            CONF.TYPES.KEY_TYPE = 'kkk'

        if selected_viewpoint == 'estimated':
            assert selected_box_mode == 'votenet'
            CONF.PATH.IMAGE = os.path.join(data_root, 'estimated-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'test-based/resnet101_features.npy')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'estimated-based/instance-masks/{}/{}.objectId.encoded.png')
            CONF.PATH.BOX = os.path.join(data_root, 'test-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'test-based/box_feat/box_feat.npy')
            CONF.PATH.VOTENET_PROJECTIONS = os.path.join(data_root, 'estimated-based/predicted_viewpoints'
                                                                    '/votenet_estimated_viewpoint_val.json')
            CONF.TYPES.KEY_TYPE = 'kk'

        if selected_viewpoint == 'topdown':
            CONF.PATH.IMAGE = os.path.join(data_root, 'topdown-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'td-based/resnet101_features.npy')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'topdown-based/instance-masks/{}/{}.vertexAttribute.encoded.png')
            CONF.PATH.BOX = os.path.join(data_root, 'td-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'td-based/box_feat/box_feat.npy')
            CONF.TYPES.KEY_TYPE = 'k'

    if selected_dataset == 'referit':
        if selected_viewpoint == 'annotated':
            CONF.PATH.IMAGE = os.path.join(data_root, 'referit-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'referit-based/resnet101_features.npy')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'referit-based/instance-masks')
            CONF.PATH.BOX = os.path.join(data_root, 'referit-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'referit-based/box_feat/box_feat.npy')
            CONF.TYPES.KEY_TYPE = 'kk'

    return CONF


def verify_visual_feat(visual_feat):
    assert ('G' in visual_feat or 'T' in visual_feat or 'C' in visual_feat)
    assert len(visual_feat) <= 3

    add_global, add_target, add_context = False, False, False
    if 'G' in visual_feat:
        add_global = True

    if 'T' in visual_feat:
        add_target = True

    if 'C' in visual_feat:
        add_context = True

    return add_global, add_target, add_context
