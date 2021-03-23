import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.DATA_ROOT = '/local-scratch/scan2cap_extracted'
CONF.PATH.CODE_ROOT = '/local-scratch/scan2cap_codebase'
CONF.PATH.OUTPUT_ROOT = os.path.join(CONF.PATH.CODE_ROOT, "outputs")
CONF.PATH.SCANNET_DIR = "/datasets/released/scannet/public/v2"
CONF.PATH.SCANS_DIR = os.path.join(CONF.PATH.SCANNET_DIR, "scans")
CONF.PATH.AGGR_JSON = os.path.join(CONF.PATH.SCANNET_DIR, "{}/{}_vh_clean.aggregation.json")  # scene_id, scene_id
CONF.PATH.SCANNET_V2_TSV = os.path.join(CONF.PATH.SCANNET_DIR, 'scannet-labels.combined.tsv')


# TODO: Here.
def get_samples(config):
    self.scene_list = [scene_id for scene_id in os.listdir(self.args.scene_list_dir) if 'scene' in scene_id]

    input_list = []
    for scene_id in self.scene_list:
        _ = [input_list.append({'scene_id': scene_id, 'object_id': f.split('-')[1].split('_')[0],
                                'ann_id': f.split('-')[1].split('_')[1].strip('.png')}) for f in
             os.listdir(os.path.join(self.scene_list_dir, scene_id)) if '.png' in f and 'thumb' not in f]

    return input_list


def get_config(exp_type, dataset, viewpoint, box):
    CONF.MODES = EasyDict()
    assert exp_type in ['nonretrieval', 'retrieval']
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
            CONF.PATH.IMAGE = os.path.join(data_root, 'annotated-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'annotated-based/resnet101_features')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'annotated-based/instance-masks')
            CONF.PATH.BOX = os.path.join(data_root, 'annotated-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'annotated-based/box_feat/box_feat.npy')

        if selected_viewpoint == 'estimated':
            assert selected_box_mode == 'votenet'
            CONF.PATH.IMAGE = os.path.join(data_root, 'estimated-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'estimated-based/resnet101_features')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'estimated-based/instance-masks')
            CONF.PATH.BOX = os.path.join(data_root, 'estimated-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'estimated-based/box_feat/box_feat.npy')
            CONF.PATH.VOTENET_PROJECTIONS = os.path.join(data_root, 'estimated-based/predicted_viewpoints'
                                                                    '/votenet_estimated_viewpoint_val.json')

        if selected_viewpoint == 'topdown':
            CONF.PATH.IMAGE = os.path.join(data_root, 'topdown-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'topdown-based/resnet101_features')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'topdown-based/instance-masks')
            CONF.PATH.BOX = os.path.join(data_root, 'topdown-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'topdown-based/box_feat/box_feat.npy')

    if selected_dataset == 'referit':
        if selected_viewpoint == 'annotated':
            CONF.PATH.IMAGE = os.path.join(data_root, 'referit-based/renders/')
            CONF.PATH.IMAGE_FEAT = os.path.join(data_root, 'referit-based/resnet101_features')
            CONF.PATH.INSTANCE_MASK = os.path.join(data_root, 'referit-based/instance-masks')
            CONF.PATH.BOX = os.path.join(data_root, 'referit-based/box/box.p')
            CONF.PATH.BOX_FEAT = os.path.join(data_root, 'referit-based/box_feat/box_feat.npy')

    return CONF
