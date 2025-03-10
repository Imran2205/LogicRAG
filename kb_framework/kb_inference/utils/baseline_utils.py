import os
import numpy as np
import base64
import logging
from datetime import datetime


def setup_logging(log_dir, model_name):
    """Set up logging with timestamp in filename"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{model_name}_qa_log_{timestamp}.txt")

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )

    return logging.getLogger()


def encode_image(image_path):
    """Encode an image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_frames_for_range(video_name, frame_range, frames_dir, num_frames=5, logger=None):
    """Get equally spaced frames from a frame range"""
    start_frame, end_frame = map(int, frame_range.split('-'))
    all_frames = list(range(start_frame, end_frame + 1))

    # Select equally spaced frames
    if len(all_frames) <= num_frames:
        selected_indices = all_frames
    else:
        # This ensures we get the start, end, and equally spaced frames in between
        selected_indices = np.linspace(0, len(all_frames) - 1, num_frames, dtype=int)
        selected_indices = [all_frames[i] for i in selected_indices]

    # Get the image paths
    frame_paths = []
    for frame_idx in selected_indices:
        frame_path = os.path.join(frames_dir, video_name, f"{frame_idx:06d}.png")
        if os.path.exists(frame_path):
            frame_paths.append(frame_path)
        else:
            if logger:
                logger.warning(f"Frame {frame_idx:06d} not found for video {video_name}")
            else:
                print(f"Warning: Frame {frame_idx:06d} not found for video {video_name}")

    return frame_paths


def convert_to_binary(answers, logger):
    """Convert 'Yes'/'No' answers to 1/0"""
    binary_answers = []

    for answer in answers:
        answer = answer.strip().lower()
        if 'yes' in answer:
            binary_answers.append(1)
        else:
            binary_answers.append(0)

    logger.info(f"Converted answers to binary: {binary_answers}")
    return binary_answers
