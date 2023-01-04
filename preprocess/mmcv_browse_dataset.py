#%%
import argparse
import os
from collections import Sequence
from pathlib import Path

import mmcv
from mmcv import Config, DictAction
from mmcv.utils import config

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset

from tqdm import tqdm
from tools.misc.browse_dataset import retrieve_data_cfg, main

#%%
args = dict(
    cfg_options=None, 
    config='../config/mask_rcnn_x101_32x4d_fpn_mstrain-poly_3x_Apollo.py', 
    not_show=True, 
    output_dir='../cache/dataset_checking', 
    # show_interval=2, 
    skip_type=['DefaultFormatBundle', 'Normalize', 'Collect']
)

cfg = retrieve_data_cfg(config_path=args['config'], skip_type=args['skip_type'], cfg_options=None)
dataset = build_dataset(cfg.data.train)

dataset.dataset



# %%
for item in tqdm(dataset):
    filename = os.path.join(args['output_dir'],
                            Path(item['filename']).name
                            ) if args['output_dir'] is not None else None

    gt_masks = item.get('gt_masks', None)
    if gt_masks is not None:
        gt_masks = mask2ndarray(gt_masks)

    imshow_det_bboxes(
        item['img'],
        item['gt_bboxes'],
        item['gt_labels'],
        gt_masks,
        class_names=dataset.CLASSES,
        show=not args['not_show'],
        # wait_time=args['show_interval'],
        out_file=filename,
        bbox_color=(255, 102, 61),
        text_color=(255, 102, 61))

# %%
