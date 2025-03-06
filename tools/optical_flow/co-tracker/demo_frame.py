#!/usr/bin/env python3
# Modified from CoTracker code to process frame sequences and output dense flow fields
import os
import torch
import argparse
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
from cotracker.predictor import CoTrackerPredictor
from natsort import natsorted

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


def load_frames_from_folder(folder_path):
    """
    Load frames from a folder containing images.
    Returns frames sorted by filename.
    """
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))

    # Sort files by name
    files = natsorted(files)

    # print(files)

    if not files:
        raise ValueError(f"No image files found in {folder_path}")

    # Read the first image to get dimensions
    sample_img = np.array(Image.open(files[0]))
    h, w = sample_img.shape[:2]

    # Pre-allocate memory for all frames
    frames = np.zeros((len(files), h, w, 3), dtype=np.uint8)

    # Load all frames
    for i, file_path in enumerate(files):
        img = np.array(Image.open(file_path))
        if len(img.shape) == 2:  # Grayscale
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]
        frames[i] = img

    return frames


def process_folder(folder_path, output_dir, model, device, dataset, sample_interval=5):
    """
    Process all consecutive frame pairs in a folder and output dense flow fields.
    Uses a sampled grid with specified interval between points to reduce memory usage.
    """
    print(f"Processing folder: {folder_path}")

    # Create output directory
    folder_name = folder_path.split('/')[-2] # os.path.basename(os.path.normpath(folder_path))
    folder_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(folder_output_dir, exist_ok=True)

    # Load all frames
    frames = load_frames_from_folder(folder_path)
    print(f"Loaded {len(frames)} frames from {folder_path}")

    # Process consecutive frame pairs
    for i in tqdm(range(len(frames) - 1), desc="Processing frame pairs"):
        # Extract current pair
        frame_pair = frames[i:i + 2]

        # Convert to tensor [1, 2, 3, H, W]
        frame_tensor = torch.from_numpy(frame_pair).permute(0, 3, 1, 2)[None].float()
        frame_tensor = frame_tensor.to(device)

        # Get frame dimensions
        _, _, _, h, w = frame_tensor.shape

        # Create a sampled grid for the first frame (every sample_interval pixels)
        y_indices = torch.arange(0, h, sample_interval, device=device)
        x_indices = torch.arange(0, w, sample_interval, device=device)
        # print(y_indices, x_indices)
        y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')

        # Reshape to [1, (H/interval)*(W/interval), 3] where the format is (t, x, y) with t=0 for first frame
        query_points = torch.stack([
            torch.zeros_like(x_grid.flatten()),  # t = 0 (first frame)
            x_grid.flatten(),  # x coordinates
            y_grid.flatten()  # y coordinates
        ], dim=-1).unsqueeze(0).type(torch.float32)

        # Track sampled pixels from first frame to second frame
        with torch.no_grad():
            pred_tracks, pred_visibility = model(
                frame_tensor,
                queries=query_points,
                grid_size=None,  # We're providing explicit query points
                grid_query_frame=0,  # Track from first frame
            )

        # Convert to numpy arrays
        # pred_tracks has shape [1, 2, N, 2] - batch, frames, points, xy
        # pred_visibility has shape [1, 2, N] - batch, frames, points
        pred_coords = pred_tracks[0].cpu().numpy()  # [2, N, 2]
        pred_vis = pred_visibility[0].cpu().numpy()  # [2, N]

        # Extract original query coordinates (from frame 0)
        query_coords_np = query_points[0, :, 1:].cpu().numpy()  # [N, 2]

        # Stack into a single array:
        # [N, 7] where each row is [query_x, query_y, pred0_x, pred0_y, pred1_x, pred1_y, visibility]
        # We're using visibility from the second frame (pred_vis[1])
        output_array = np.column_stack([
            query_coords_np,  # [N, 2] - original x,y coordinates
            pred_coords[0],  # [N, 2] - frame 0 predictions (should match queries)
            pred_coords[1],  # [N, 2] - frame 1 predictions
            pred_vis[1].reshape(-1, 1)  # [N, 1] - visibility in frame 1
        ])

        # Save as numpy array
        if dataset == 'KITTI':
            output_path = os.path.join(folder_output_dir, f"flow-{folder_name}_{i + 1:06d}.npy")
        elif dataset == 'CARLA':
            output_path = os.path.join(folder_output_dir, f"flow-{folder_name}_{i + 2:05d}.npy")

        np.save(output_path, output_array)

    print(f"Completed processing {folder_path}")
    return len(frames) - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent_folder",
        required=True,
        help="Parent folder containing subfolders of image sequences",
    )
    parser.add_argument(
        "--output_dir",
        default="./flow_outputs",
        help="Directory to save flow fields",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters (optional)",
    )
    parser.add_argument(
        "--dataset",
        default='CARLA',
        help="Name of the dataset",
    )
    parser.add_argument(
        "--use_v2_model",
        action="store_true",
        help="Use CoTracker2 instead of CoTracker3",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=10,
        help="Sample every Nth pixel (to reduce memory usage)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize the model
    if args.checkpoint is not None:
        if args.use_v2_model:
            model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=True)
        else:
            # Use offline mode with window_len=2 for consecutive frames
            model = CoTrackerPredictor(
                checkpoint=args.checkpoint,
                v2=False,
                offline=True,
                window_len=60,  # Just process pairs of frames
            )
    else:
        # Load the latest CoTracker3 model with window_len=2
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        # Set window length to 2 for processing only consecutive frames
        # model.window_len = 60

    model = model.to(DEFAULT_DEVICE)
    print(f"Model loaded on {DEFAULT_DEVICE}")

    # Get all subfolders in the parent folder
    subfolders = natsorted([f.path+'/rgb' for f in os.scandir(args.parent_folder) if f.is_dir()])

    # print(subfolders)

    if not subfolders:
        print(f"No subfolders found in {args.parent_folder}")
        exit(1)

    print(f"Found {len(subfolders)} video folders to process")

    # Process each subfolder
    total_processed_pairs = 0
    for folder in subfolders:
        pairs_processed = process_folder(folder, args.output_dir, model, DEFAULT_DEVICE,
                                         dataset=args.dataset, sample_interval=args.sample_interval)
        total_processed_pairs += pairs_processed

    print(f"Total processed {total_processed_pairs} frame pairs across {len(subfolders)} folders")
    print(f"Flow fields saved to {args.output_dir}")
