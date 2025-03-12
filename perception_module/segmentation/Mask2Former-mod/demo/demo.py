# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from PIL import Image

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo


# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False


def panoptic_to_rgb_mask(panoptic_tensor, segments_info):
    """
    Convert panoptic segmentation to RGB mask where:
    - R channel: category_id (object class)
    - G,B channels: segment ID (split into two 8-bit channels)

    Args:
        panoptic_tensor: tensor of shape (H, W) with segment IDs
        segments_info: list of dictionaries containing segment information
            Each dict contains:
            - 'id': segment ID
            - 'isthing': boolean indicating if it's a thing or stuff class
            - 'category_id': class ID
            - 'area': area of the segment

    Returns:
        rgb_mask: numpy array of shape (H, W, 3) with RGB values
    """
    # Get dimensions
    H, W = panoptic_tensor.shape

    # Initialize empty RGB mask
    rgb_mask = np.zeros((H, W, 3), dtype=np.uint8)

    # Convert tensor to CPU if on GPU
    panoptic_array = panoptic_tensor.cpu().numpy()

    # Create a mapping from segment IDs to category IDs
    id_to_category = {seg['id']: seg['category_id'] for seg in segments_info}

    # Process each pixel - we'll optimize it by processing unique IDs
    unique_ids = np.unique(panoptic_array)

    for segment_id in unique_ids:
        if segment_id == 0:  # Sometimes 0 is used for void/ignore regions
            continue

        # Find the category ID for this segment
        category_id = id_to_category.get(segment_id, 0)

        # Create a mask for this segment
        mask = (panoptic_array == segment_id)

        # Assign RGB values
        rgb_mask[mask, 0] = category_id  # R channel: category ID

        # Split segment ID into G and B channels (16 bits total)
        g_value = (segment_id >> 8) & 255  # High byte
        b_value = segment_id & 255  # Low byte

        rgb_mask[mask, 1] = g_value
        rgb_mask[mask, 2] = b_value

    return rgb_mask


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            ins_rgb = Image.fromarray(
                panoptic_to_rgb_mask(predictions["panoptic_seg"][0], predictions["panoptic_seg"][1])
            )

            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )

            if args.output:
                vid_name = path.split('/')[-2]
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    if not os.path.exists(os.path.join(args.output, vid_name, 'combined')):
                        os.makedirs(os.path.join(args.output, vid_name, 'combined'))

                    out_filename = os.path.join(
                        args.output, vid_name, 'combined',
                        f"{vid_name}_{os.path.basename(path)}"
                    )

                    if not os.path.exists(os.path.join(args.output, vid_name, 'label')):
                        os.makedirs(os.path.join(args.output, vid_name, 'label'))

                    ins_rgb_path = os.path.join(
                        args.output, vid_name, 'label',
                        f"{vid_name}_{os.path.basename(path)}"
                    )
                    ins_rgb.save(ins_rgb_path)
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
    elif args.webcam:
        assert args.input is None, "Cannot have both --input and --webcam!"
        assert args.output is None, "output not yet supported with --webcam!"
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cam.release()
        cv2.destroyAllWindows()
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        codec, file_ext = (
            ("x264", ".mkv") if test_opencv_video_format("x264", ".mkv") else ("mp4v", ".mp4")
        )
        if codec == ".mp4v":
            warnings.warn("x264 codec not available, switching to mp4v")
        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + file_ext
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.video_input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
