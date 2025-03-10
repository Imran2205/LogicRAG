# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

PFB_SEM_SEG_CATEGORIES = {
    0:  {'name': 'unlabeled', 'color': (0, 0, 0), 'train_id': 255},
    1:  {'name': 'ambiguous', 'color': (111, 74, 0), 'train_id': 255},
    2:  {'name': 'sky', 'color': (70, 130, 180), 'train_id': 0},
    3:  {'name': 'road', 'color': (128, 64, 128), 'train_id': 1},
    4:  {'name': 'sidewalk', 'color': (244, 35, 232), 'train_id': 2},
    5:  {'name': 'rail track', 'color': (230, 150, 140), 'train_id': 255},
    6:  {'name': 'terrain', 'color': (152, 251, 152), 'train_id': 3},
    7:  {'name': 'tree', 'color': (87, 182, 35), 'train_id': 4},
    8:  {'name': 'vegetation', 'color': (35, 142, 35), 'train_id': 5},
    9:  {'name': 'building', 'color': (70, 70, 70), 'train_id': 6},
    10: {'name': 'infrastructure', 'color': (153, 153, 153), 'train_id': 7},
    11: {'name': 'fence', 'color': (190, 153, 153), 'train_id': 8},
    12: {'name': 'billboard', 'color': (150, 20, 20), 'train_id': 9},
    13: {'name': 'trafficlight', 'color': (250, 170, 30), 'train_id': 10},
    14: {'name': 'traffic sign', 'color': (220, 220, 0), 'train_id': 11},
    15: {'name': 'mobile barrier', 'color': (180, 180, 100), 'train_id': 12},
    16: {'name': 'fire hydrant', 'color': (173, 153, 153), 'train_id': 13},
    17: {'name': 'chair', 'color': (168, 153, 153), 'train_id': 14},
    18: {'name': 'trash', 'color': (81, 0, 21), 'train_id': 15},
    19: {'name': 'trashcan', 'color': (81, 0, 81), 'train_id': 16},
    20: {'name': 'person', 'color': (220, 20, 60), 'train_id': 17},
    21: {'name': 'animal', 'color': (255, 0, 0), 'train_id': 255},
    22: {'name': 'bicycle', 'color': (119, 11, 32), 'train_id': 255},
    23: {'name': 'motorcycle', 'color': (0, 0, 230), 'train_id': 18},
    24: {'name': 'car', 'color': (0, 0, 142), 'train_id': 19},
    25: {'name': 'van', 'color': (0, 80, 100), 'train_id': 20},
    26: {'name': 'bus', 'color': (0, 60, 100), 'train_id': 21},
    27: {'name': 'truck', 'color': (0, 0, 70), 'train_id': 22},
    28: {'name': 'trailer', 'color': (0, 0, 90), 'train_id': 255},
    29: {'name': 'train', 'color': (0, 80, 100), 'train_id': 255},
    30: {'name': 'plane', 'color': (0, 100, 100), 'train_id': 255},
    31: {'name': 'boat', 'color': (50, 0, 90), 'train_id': 255},
}


def _get_pfb_meta():
    stuff_classes = [PFB_SEM_SEG_CATEGORIES[k]["name"] for k in PFB_SEM_SEG_CATEGORIES.keys() if PFB_SEM_SEG_CATEGORIES[k]["train_id"] != 255]
    assert len(stuff_classes) == 23

    stuff_colors = [list(PFB_SEM_SEG_CATEGORIES[k]["color"]) for k in PFB_SEM_SEG_CATEGORIES.keys() if PFB_SEM_SEG_CATEGORIES[k]["train_id"] != 255]
    assert len(stuff_colors) == 23

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_pfb(root):
    root = os.path.join(root, "data-uws-color/data_for_mask_2_former")
    meta = _get_pfb_meta()
    for name, dirname in [("train", "train"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"pfb_sem_seg_{name}"
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


_root = "/kaggle/input/"
register_all_pfb(_root)
