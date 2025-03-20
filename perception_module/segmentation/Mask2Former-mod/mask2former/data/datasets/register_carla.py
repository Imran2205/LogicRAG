# Copyright (c) Facebook, Inc. and its affiliates.
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

CARLA_SEM_SEG_CATEGORIES = {
    0: {'name': 'unlabeled', 'color': (0, 0, 0), 'train_id': 0, 'id': 255},
    1: {'name': 'building', 'color': (70, 70, 70), 'train_id': 1, 'id': 0},
    2: {'name': 'fence', 'color': (100, 40, 40), 'train_id': 2, 'id': 1},
    3: {'name': 'other', 'color': (55, 90, 80), 'train_id': 3, 'id': 255},
    4: {'name': 'pedestrian', 'color': (220, 20, 60), 'train_id': 4, 'id': 2},
    5: {'name': 'pole', 'color': (153, 153, 153), 'train_id': 5, 'id': 3},
    6: {'name': 'roadline', 'color': (157, 234, 50), 'train_id': 6, 'id': 4},
    7: {'name': 'road', 'color': (128, 64, 128), 'train_id': 7, 'id': 5},
    8: {'name': 'sidewalk', 'color': (244, 35, 232), 'train_id': 8, 'id': 6},
    9: {'name': 'vegetation', 'color': (107, 142, 35), 'train_id': 9, 'id': 7},
    10: {'name': 'vehicles', 'color': (0, 0, 142), 'train_id': 10, 'id': 8},
    11: {'name': 'wall', 'color': (102, 102, 156), 'train_id': 11, 'id': 9},
    12: {'name': 'trafficsign', 'color': (220, 220, 0), 'train_id': 12, 'id': 10},
    13: {'name': 'sky', 'color': (70, 130, 180), 'train_id': 13, 'id': 11},
    14: {'name': 'ground', 'color': (81, 0, 81), 'train_id': 14, 'id': 12},
    15: {'name': 'bridge', 'color': (150, 100, 100), 'train_id': 15, 'id': 13},
    16: {'name': 'railtrack', 'color': (230, 150, 140), 'train_id': 16, 'id': 14},
    17: {'name': 'guardrail', 'color': (180, 165, 180), 'train_id': 17, 'id': 15},
    18: {'name': 'trafficlight', 'color': (250, 170, 30), 'train_id': 18, 'id': 16},
    19: {'name': 'static', 'color': (110, 190, 160), 'train_id': 19, 'id': 17},
    20: {'name': 'dynamic', 'color': (170, 120, 50), 'train_id': 20, 'id': 18},
    21: {'name': 'water', 'color': (45, 60, 150), 'train_id': 21, 'id': 19},
    22: {'name': 'terrain', 'color': (145, 170, 100), 'train_id': 22, 'id': 20}
}


def _get_carla_meta():
    stuff_classes = [CARLA_SEM_SEG_CATEGORIES[k]["name"] for k in CARLA_SEM_SEG_CATEGORIES.keys() if CARLA_SEM_SEG_CATEGORIES[k]["id"] != 255]
    assert len(stuff_classes) == 21

    stuff_colors = [list(CARLA_SEM_SEG_CATEGORIES[k]["color"]) for k in CARLA_SEM_SEG_CATEGORIES.keys() if CARLA_SEM_SEG_CATEGORIES[k]["id"] != 255]
    assert len(stuff_colors) == 21

    ret = {
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    return ret


def register_all_carla(root):
    root = os.path.join(root, "carla_semantic")
    meta = _get_carla_meta()
    for name, dirname in [("train", "train"), ("val", "validation")]:
        image_dir = os.path.join(root, dirname, "images")
        gt_dir = os.path.join(root, dirname, "labels")
        name = f"carla_sem_seg_{name}"
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


_root = "/kaggle/working/"
register_all_carla(_root)
