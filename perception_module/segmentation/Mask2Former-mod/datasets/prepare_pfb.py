import glob
import os
import shutil
from tqdm import tqdm
from PIL import Image
import numpy as np

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


dataset_root = ''
dataset_out = ''


def copy_files(state):
    train_image_files = glob.glob(
        os.path.join(
            f'{dataset_root}/{state}/images',
            f'*/*/*/*', f'*.png')
    )

    temp_label_files = glob.glob(
        os.path.join(
            f'{dataset_root}/{state}/labels',
            f'*/*/*/*', f'*.png')
    )

    dataset_out_train_images = os.path.join(dataset_out, f'{state}/images')
    dataset_out_train_labels = os.path.join(dataset_out, f'{state}/labels')
    dataset_out_train_labels_color = os.path.join(dataset_out, f'{state}/labels_color')

    if not os.path.exists(dataset_out_train_images):
        os.makedirs(dataset_out_train_images)
    if not os.path.exists(dataset_out_train_labels):
        os.makedirs(dataset_out_train_labels)
    if not os.path.exists(dataset_out_train_labels_color):
        os.makedirs(dataset_out_train_labels_color)

    for lbl in tqdm(temp_label_files):
        shutil.copy(lbl, dataset_out_train_labels_color)
        lbl_img = Image.open(lbl).convert('RGB')
        lbl_img = np.array(lbl_img)
        lbl_img_updated = np.ones(lbl_img.shape[:2]) * 255
        for key in PFB_SEM_SEG_CATEGORIES.keys():
            indices = np.argwhere(np.all(lbl_img == PFB_SEM_SEG_CATEGORIES[key]['color'], axis=-1))
            rr, cc = indices[:, 0], indices[:, 1]
            lbl_img_updated[rr, cc] = PFB_SEM_SEG_CATEGORIES[key]['train_id']
        lbl_img_updated = lbl_img_updated.astype(np.uint8)
        lbl_img_updated = Image.fromarray(lbl_img_updated, 'L')
        lbl_img_updated.save(
            os.path.join(
                dataset_out_train_labels,
                os.path.basename(lbl)
            )
        )

    for img in tqdm(train_image_files):
        shutil.copy(img, dataset_out_train_images)


folders = ['train', 'val']

for fol in folders:
    copy_files(fol)