from PIL import Image
import numpy as np
import plotly.express as px
import cv2
import png
import io

# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels/Nettles_19.png"
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels/Crab_18.png"
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels/Dolphin_24.png"
image = "/Volumes/ssd_imran/carla_dataset/testing/tracker_input/gt_tracker/depth_gray/town_10_00029058.tiff"
#
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels_old/Nettles_19.png"
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels_old/Crab_18.png"
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels_old/Dolphin_24.png"
# image = "/Users/imrankabir/Desktop/research/few_shot/underwater_few_shot/extras/data_for_mask_2_former/train/labels_old/Whale_12.png"

def save_image(arr, filename):
    # is_success, buffer = cv2.imencode(".tiff", arr)
    # io_buf = io.BytesIO(buffer)
    # print(type(io_buf))
    # with open(filename, 'wb') as f:
    #     f.write(io_buf.read())
    cv2.imwrite(filename, arr)


def raw_depth_to_gray_32bit(raw_depth, verbose=False):
    depth = raw_depth.convert('RGB')
    depth = np.array(depth)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    depth = np.dot(depth[..., :3], [1, 256, 256 * 256]).astype(np.int32)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    return depth


def raw_image(image_, verbose=False):
    depth = np.array(image_)

    if verbose:
        print(np.max(depth), np.min(depth), depth.shape, depth.dtype)

    return depth


img = Image.open(image)

# img = raw_depth_to_gray_32bit(img, verbose=True)
img = raw_image(img, verbose=True)

# save_image(img, filename='demo.tiff')

fig = px.imshow(img)
fig.show()
