# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

UWS_SEM_SEG_CATEGORIES = {
    0:  {'name': 'unlabeled',   'train_id': 255, 'color': (0,   0,   0)},
    1:  {'name': 'crab',        'train_id': 0,   'color': (128, 64,  128)},
    2:  {'name': 'crocodile',   'train_id': 1,   'color': (244, 35,  232)},
    3:  {'name': 'dolphin',     'train_id': 2,   'color': (70,  70,  70)},
    4:  {'name': 'frog',        'train_id': 3,   'color': (102, 102, 156)},
    5:  {'name': 'nettles',     'train_id': 4,   'color': (190, 153, 153)},
    6:  {'name': 'octopus',     'train_id': 5,   'color': (153, 153, 153)},
    7:  {'name': 'otter',       'train_id': 6,   'color': (250, 170, 30)},
    8:  {'name': 'penguin',     'train_id': 7,   'color': (220, 220, 0)},
    9:  {'name': 'polar_bear',  'train_id': 8,   'color': (107, 142, 35)},
    10: {'name': 'sea_anemone', 'train_id': 9,   'color': (152, 251, 152)},
    11: {'name': 'sea_urchin',  'train_id': 10,  'color': (70,  130, 180)},
    12: {'name': 'seahorse',    'train_id': 11,  'color': (220, 20,  60)},
    13: {'name': 'seal',        'train_id': 12,  'color': (253, 0,   0)},
    14: {'name': 'shark',       'train_id': 13,  'color': (0,   0,   142)},
    15: {'name': 'shrimp',      'train_id': 14,  'color': (0,   0,   70)},
    16: {'name': 'star_fish',   'train_id': 15,  'color': (0,   60,  100)},
    17: {'name': 'stingray',    'train_id': 16,  'color': (0,   80,  100)},
    18: {'name': 'squid',       'train_id': 17,  'color': (0,   0,   230)},
    19: {'name': 'turtle',      'train_id': 18,  'color': (119, 11,  32)},
    20: {'name': 'whale',       'train_id': 19,  'color': (111, 74,  0)},
    21: {'name': 'nudibranch',  'train_id': 20,  'color': (81,  0,   81)},
    22: {'name': 'coral',       'train_id': 21,  'color': (250, 170, 160)},
    23: {'name': 'rock',        'train_id': 22,  'color': (230, 150, 140)},
    24: {'name': 'water',       'train_id': 23,  'color': (180, 165, 180)},
    25: {'name': 'sand',        'train_id': 24,  'color': (150, 100, 100)},
    26: {'name': 'plant',       'train_id': 25,  'color': (150, 120, 90)},
    27: {'name': 'human',       'train_id': 26,  'color': (153, 153, 153)},
    28: {'name': 'reef',        'train_id': 27,  'color': (0,   0,   110)},
    29: {'name': 'others',      'train_id': 28,  'color': (47,  220, 70)}
}


def _get_uws_meta():
    stuff_classes = [UWS_SEM_SEG_CATEGORIES[k]["name"] for k in UWS_SEM_SEG_CATEGORIES.keys() if UWS_SEM_SEG_CATEGORIES[k]["name"] != 'unlabeled']
    assert len(stuff_classes) == 29

    stuff_colors = [list(UWS_SEM_SEG_CATEGORIES[k]["color"]) for k in UWS_SEM_SEG_CATEGORIES.keys() if UWS_SEM_SEG_CATEGORIES[k]["name"] != 'unlabeled']
    assert len(stuff_colors) == 29

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_uws(root):
    root = os.path.join(root, "uw_sem_seg_dataset")
    meta = _get_uws_meta()
    for name, dirname in [("train", "train"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"uws_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="png")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,  # different from other datasets, Mapillary Vistas sets ignore_label to 65
            **meta,
        )

        # print(f"{image_dir}  <<<>>> {gt_dir}")

    # print("done")


_root = "/content/"
register_all_uws(_root)
