#!/usr/bin/env python3
import os
import numpy as np
import cv2
import glob
from tqdm import tqdm
import argparse
import math
from PIL import Image
from natsort import natsorted

# Static class IDs (modify based on your segmentation labels)
STATIC_CLASSES = [2, 3, 5, 6, 7, 8]

DATASET = 'KITTI'
FX_DEPTH = 721.5377
CX_DEPTH = 609.5593
FY_DEPTH = 721.5377
CY_DEPTH = 172.854
SCALE = 256.0


def set_camera_intrinsic():
    global FX_DEPTH, CX_DEPTH, FY_DEPTH, CY_DEPTH, SCALE
    if DATASET == 'KITTI':
        # Camera intrinsics for KITTI
        FX_DEPTH = 721.5377
        CX_DEPTH = 609.5593
        FY_DEPTH = 721.5377
        CY_DEPTH = 172.854
        SCALE = 256.0
    elif DATASET == 'CARLA':
        # Camera intrinsics for CARLA
        FX_DEPTH = 168.05
        FY_DEPTH = 168.05
        CX_DEPTH = 480/2
        CY_DEPTH = 270/2
        SCALE = 65.536


def load_segmentation_map(path):
    """Load segmentation map with class IDs"""
    if os.path.exists(path):
        seg_map = np.array(Image.open(path))
        # If the segmentation map has multiple channels, use only the first one for class IDs
        if len(seg_map.shape) > 2:
            return seg_map[:, :, 0]
        return seg_map
    return None


def load_depth_map(path, img_shape):
    target_h, target_w = img_shape

    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth.dtype == np.uint16:
        depth = depth.astype(np.float32) / SCALE

    depth_h, depth_w = depth.shape[:2]

    if depth_h == target_h and depth_w == target_w:
        # No padding needed
        return depth

    # Calculate padding amounts
    pad_top = (target_h - depth_h) // 2
    pad_bottom = target_h - depth_h - pad_top
    pad_left = (target_w - depth_w) // 2
    pad_right = target_w - depth_w - pad_left

    # Pad the depth map with zeros (representing invalid/unknown depth)
    padded_depth = np.pad(
        depth,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0.
    )

    return padded_depth


def load_flow_data(path):
    """Load optical flow data"""
    if os.path.exists(path):
        try:
            flow_data = np.load(path)
            return flow_data
        except Exception as e:
            print(f"Error loading flow data from {path}: {e}")
    return None


def pixel_to_3d(x, y, depth):
    """Convert pixel coordinates and depth to 3D world coordinates"""
    x_3d = (x - CX_DEPTH) * depth / FX_DEPTH
    y_3d = (y - CY_DEPTH) * depth / FY_DEPTH
    z_3d = depth
    return [x_3d, y_3d, z_3d]


def process_video_frames(video_folder, output_dir, distance_threshold=2.0, max_depth=50.0, sample_interval=20):
    """
    Process all consecutive frame pairs in a video to find static points.

    Args:
        video_folder: Path to the folder containing the video frames
        output_dir: Directory to save keypoint files
        distance_threshold: Maximum allowed 3D distance for a point to be considered static
        max_depth: Maximum allowed depth value
        sample_interval: Sample every Nth pixel to reduce processing time
    """
    # Get paths to data directories
    segmentation_dir = os.path.join(video_folder, "label")
    depth_dir = os.path.join(video_folder, "depth")
    flow_dir = os.path.join(video_folder, "flow")

    # Get video name
    video_name = os.path.basename(video_folder)

    # Create output directory for this video
    video_output_dir = os.path.join(output_dir, video_name, 'keypoints')
    os.makedirs(video_output_dir, exist_ok=True)

    # Find all segmentation files
    seg_files = sorted(glob.glob(os.path.join(segmentation_dir, "*.png")))

    if not seg_files:
        print(f"No segmentation files found in {segmentation_dir}")
        return

    # Process consecutive frame pairs
    total_keypoints = 0

    for i in tqdm(range(len(seg_files) - 1), desc=f"Processing {video_name}"):
        # Get frame numbers
        frame1_name = os.path.basename(seg_files[i]).split('.')[0]
        frame2_name = os.path.basename(seg_files[i + 1]).split('.')[0]

        try:
            frame1_num = int(frame1_name.split('_')[-1])
            frame2_num = int(frame2_name.split('_')[-1])
        except ValueError:
            # Handle different naming conventions
            frame1_num = i
            frame2_num = i + 1

        # Load segmentation maps
        seg_map1 = load_segmentation_map(seg_files[i])
        seg_map2 = load_segmentation_map(seg_files[i + 1])

        if seg_map1 is None or seg_map2 is None:
            print(f"Skipping frames {frame1_num}-{frame2_num}: Missing segmentation maps")
            continue

        # Load depth maps
        depth_map1 = load_depth_map(os.path.join(depth_dir, f"{frame1_name}.png"), seg_map1.shape)
        depth_map2 = load_depth_map(os.path.join(depth_dir, f"{frame2_name}.png"), seg_map2.shape)

        assert depth_map1.shape == seg_map1.shape == depth_map2.shape == seg_map2.shape

        if depth_map1 is None or depth_map2 is None:
            print(f"Skipping frames {frame1_num}-{frame2_num}: Missing depth maps")
            continue

        # Load flow data
        if DATASET == 'KITTI':
            flow_file = os.path.join(flow_dir, f"flow-{video_name}_{frame2_num:06d}.npy")
        else:
            flow_file = os.path.join(flow_dir, f"flow-{video_name}_{frame2_num:05d}.npy")
        flow_data = load_flow_data(flow_file)

        if flow_data is None:
            print(f"Skipping frames {frame1_num}-{frame2_num}: Missing flow data")
            continue

        # Create a mask of static object classes
        static_mask1 = np.isin(seg_map1, STATIC_CLASSES)

        # Sample points from static regions
        h, w = static_mask1.shape

        # Create a grid of sample points
        y_indices = np.arange(20, h - 20, sample_interval)
        x_indices = np.arange(20, w - 20, sample_interval)

        # Process each sampled point
        frame_static_points = []

        for y in y_indices:
            for x in x_indices:
                # Skip if not in static region
                if not static_mask1[y, x]:
                    continue

                # Get class ID
                class_id = seg_map1[y, x]

                # Find closest point in flow data
                distances = np.sqrt((flow_data[:, 0] - x) ** 2 + (flow_data[:, 1] - y) ** 2)
                closest_idx = np.argmin(distances)

                # Get the corresponding point in frame 2
                x2, y2 = flow_data[closest_idx, 4:6]

                # Convert to integers (for depth lookup)
                x2_int, y2_int = int(round(x2)), int(round(y2))

                # Skip if out of bounds
                if not (20 <= y2_int < h - 20 and 20 <= x2_int < w - 20):
                    continue

                # Check if the point is also in a static region in frame 2
                if not np.isin(seg_map2[y2_int, x2_int], STATIC_CLASSES):
                    continue

                # Get depths
                depth1 = depth_map1[y, x]
                depth2 = depth_map2[y2_int, x2_int]

                # Skip if depths are invalid or too large
                if depth1 > max_depth or depth2 > max_depth:
                    continue

                # Convert to 3D coordinates
                point3d_1 = pixel_to_3d(x, y, depth1)
                point3d_2 = pixel_to_3d(x2_int, y2_int, depth2)

                # Check if the point is static (hasn't moved much)
                dist = math.dist(point3d_1, point3d_2)
                if dist <= distance_threshold:
                    # Store the point pair
                    frame_static_points.append([int(x), int(y), depth1, int(round(x2)), int(round(y2)), depth2])

        # Save keypoints for this frame pair
        frame_static_points = np.array(frame_static_points)
        if len(frame_static_points) > 0:
            # Create output filename based on the second frame number
            if DATASET == 'KITTI':
                output_file = os.path.join(video_output_dir, f"keypoints-{video_name}_{frame2_num:06d}.npy")
            else:
                output_file = os.path.join(video_output_dir, f"keypoints-{video_name}_{frame2_num:05d}.npy")
            np.save(output_file, frame_static_points)
            total_keypoints += len(frame_static_points)

    print(f"Saved a total of {total_keypoints} keypoints for {video_name} across {len(seg_files) - 1} frame pairs")


def main():
    global DATASET
    parser = argparse.ArgumentParser(description="Extract static points from video frames")
    parser.add_argument("--data_dir", required=True, help="Parent directory containing semantic labels, depth and optical flow")
    parser.add_argument("--output_dir", required=True, help="Directory to save keypoints")
    parser.add_argument("--distance_threshold", type=float, default=0.7,
                        help="Maximum 3D distance for a point to be considered static")
    parser.add_argument("--max_depth", type=float, default=50.0,
                        help="Maximum allowed depth value")
    parser.add_argument("--sample_interval", type=int, default=20,
                        help="Sample every Nth pixel")
    parser.add_argument("--dataset", default='CARLA', help="Name of the dataset")

    args = parser.parse_args()

    DATASET = args.dataset.upper()

    set_camera_intrinsic()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all video directories
    video_dirs = natsorted([f.path for f in os.scandir(args.data_dir) if f.is_dir()])

    if not video_dirs:
        print(f"No video directories found in {args.data_dir}")
        return

    # Process each video
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)

        process_video_frames(
            video_dir,
            args.output_dir,
            distance_threshold=args.distance_threshold,
            max_depth=args.max_depth,
            sample_interval=args.sample_interval
        )

    print(f"Processed {len(video_dirs)} videos")


if __name__ == "__main__":
    main()
