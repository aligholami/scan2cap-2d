import os
from easydict import EasyDict

CONF = EasyDict()
CONF.PATH = EasyDict()
CONF.PATH.DATA_ROOT = '/local-scratch/scan2cap_extracted'
CONF.PATH.CODE_ROOT = '/local-scratch/scan2cap_codebase'
CONF.PATH.OUTPUT_ROOT = os.path.join(CONF.PATH.CODE_ROOT, "outputs")
CONF.PATH.SCANNET_DIR = "/datasets/released/scannet/public/v2/scans"


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
