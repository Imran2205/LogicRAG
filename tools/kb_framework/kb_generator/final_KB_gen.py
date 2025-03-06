#!/usr/bin/env python3
import os
import sys
import pickle
import numpy as np
import argparse
from collections import defaultdict
import math
from scipy import stats
import aima.utils
import aima.logic
from natsort import natsorted
from tqdm import tqdm
import multiprocessing
from functools import partial

# Label dictionary mapping class IDs to names
LABELS_DICT = {
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

DATASET = 'CARLA'
VEHICLE_TYPE_MAPPING = {
    'car': 'Car',
    'truck': 'Truck',
    'bus': 'Bus',
    'motorcycle': 'Motorcycle',
    'van': 'Van'
}

# Constants for position determination
DISTANCE_THRESHOLD = 50  # Z distance threshold for Near/Far
SPEED_THRESHOLD_VEHICLE = 0.7  # Threshold for Moving/NotMoving for vehicles
SPEED_THRESHOLD_PEDESTRIAN = 0.3  # Threshold for Moving/NotMoving for pedestrians
VEL_SLOPE_THRESHOLD = 0.5  # Threshold for SpeedUp/SpeedDown
DIST_SLOPE_THRESHOLD = 0.5  # Threshold for DistanceIncrease/DistanceDecrease
VERY_CLOSE_DISTANCE = 0.3  # Threshold for TooClose predicate
ZERO_DISTANCE_THRESHOLD = 0.1  # Threshold for DistanceZeroStart/DistanceZeroEnd


def load_tracker_data(tracker_dir, video_name):
    """
    Load tracker data from pickle files for a specific video

    Args:
        tracker_dir: Directory containing tracker pickle files
        video_name: Name of the video to process

    Returns:
        Dictionary of track data by class
    """
    track_data = {}

    # Find all pickle files for this video
    video_dir = os.path.join(tracker_dir, video_name)
    if not os.path.exists(video_dir):
        print(f"Error: Video directory {video_dir} not found")
        return {}

    pickle_files = [f for f in os.listdir(video_dir) if f.endswith('.pickle') and f.startswith('trackers_')]

    for pkl_file in pickle_files:
        class_name = pkl_file.replace(f'trackers_{video_name}_', '').replace('.pickle', '')
        file_path = os.path.join(video_dir, pkl_file)

        with open(file_path, 'rb') as f:
            class_data = pickle.load(f)
            track_data[class_name] = class_data

    return track_data


def get_frames_in_window(track_data, window_start, window_size=10):
    """
    Get all frames within a window range

    Args:
        track_data: Dictionary of track data by class
        window_start: Starting frame number
        window_size: Number of frames in window

    Returns:
        List of frame numbers in the window
    """
    window_end = window_start + window_size - 1
    all_frames = set()

    for class_name, class_data in track_data.items():
        for track_id, track_info in class_data.items():
            for frame_num in track_info.keys():
                frame_idx = int(frame_num)
                if window_start <= frame_idx <= window_end:
                    all_frames.add(frame_idx)

    return natsorted(list(all_frames))


def get_objects_in_window(track_data, window_frames):
    """
    Get all objects present in a window

    Args:
        track_data: Dictionary of track data by class
        window_frames: List of frame numbers in the window

    Returns:
        Dictionary mapping object IDs to their class and frames
    """
    objects = {}

    for class_name, class_data in track_data.items():
        for track_id, track_info in class_data.items():
            object_id = f"{class_name.capitalize()}_{track_id}"

            object_frames = []
            all_frames = []
            for frame_num, frame_data in track_info.items():
                frame_idx = int(frame_num)
                all_frames.append(frame_idx)
                if frame_idx in window_frames:
                    object_frames.append(frame_idx)

            if object_frames:
                objects[object_id] = {
                    'class': class_name,
                    'frames': natsorted(object_frames),
                    'all_frames': natsorted(all_frames),
                    'track_data': track_info
                }

    return objects


def get_position_category(x, z, max_x, max_z=None):
    """
    Determine position based on x and z coordinates

    Args:
        x: X coordinate (horizontal position)
        z: Z coordinate (depth)
        max_x: Maximum X value for normalization
        max_z: Maximum Z value (optional)

    Returns:
        Tuple of (x_position, z_position) where:
        - x_position is 'left', 'front', or 'right'
        - z_position is 'near' or 'far'
    """
    # Normalize x coordinate to [0, 1]

    norm_x = x / max_x

    # Determine horizontal position
    if norm_x < 0.4:
        x_position = 'left'
    elif norm_x > 0.6:
        x_position = 'right'
    else:
        x_position = 'front'

    # Determine depth position
    if z < DISTANCE_THRESHOLD:
        z_position = 'near'
    else:
        z_position = 'far'

    return (x_position, z_position)


def filter_outliers(data_list):
    """
    Filter outliers from a list of values and smooth the result

    Args:
        data_list: List of numeric values

    Returns:
        Filtered and smoothed list of values
    """
    import numpy as np
    from scipy import stats
    from scipy.signal import savgol_filter

    if not data_list or len(data_list) < 3:
        return data_list

    # Step 1: Replace outliers with more reasonable values
    # Calculate Z-scores
    z_scores = stats.zscore(data_list)

    # Replace outliers with mean of non-outliers
    filtered = data_list.copy()
    outlier_indices = np.where(np.abs(z_scores) > 2.0)[0]

    if len(outlier_indices) > 0 and len(outlier_indices) < len(data_list) - 1:
        # Calculate mean of non-outliers
        non_outlier_indices = np.where(np.abs(z_scores) <= 2.0)[0]
        mean_value = np.mean([data_list[i] for i in non_outlier_indices])

        # Replace outliers with mean
        for idx in outlier_indices:
            filtered[idx] = mean_value

    # Step 2: Apply Savitzky-Golay filter for smoothing
    # The window size needs to be odd and less than the data length
    if len(filtered) >= 5:
        if len(filtered) % 2 == 0:
            # For even-length data, use next smallest odd number
            window_length = min(len(filtered) - 1, 9)
        else:
            # For odd-length data, use the length directly
            window_length = min(len(filtered), 9)
        # print(window_length, len(filtered))
        poly_order = min(window_length - 1, 3)  # Polynomial order must be less than window_length

        try:
            smoothed = savgol_filter(filtered, window_length, poly_order)
            return smoothed.tolist()
        except Exception as e:
            print(f"Savgol filtering failed: {e}, returning outlier-filtered data")
            return filtered

    # If too few points for Savgol, use simple moving average for smoothing
    elif len(filtered) >= 3:
        window_size = 3
        smoothed = np.convolve(filtered, np.ones(window_size) / window_size, mode='same')
        return smoothed.tolist()

    return filtered


def calculate_slope(values):
    """
    Calculate slope of values

    Args:
        values: List of numeric values

    Returns:
        Slope of values or 0 if not enough data
    """
    if not values or len(values) < 2:
        return 0

    # Use linear regression to calculate slope
    x = np.arange(len(values))
    y = np.array(values)

    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def calculate_distances(objects, window_frames):
    """
    Calculate distances between all pairs of objects in a window

    Args:
        objects: Dictionary of objects in the window
        window_frames: List of frame numbers in the window

    Returns:
        Dictionary mapping object pairs to distance lists
    """
    distances = {}

    object_ids = list(objects.keys())

    # Calculate distances for all pairs of objects
    for i in range(len(object_ids)):
        obj1_id = object_ids[i]
        obj1 = objects[obj1_id]

        for j in range(i + 1, len(object_ids)):
            obj2_id = object_ids[j]
            obj2 = objects[obj2_id]

            # Find common frames
            common_frames = natsorted(set(obj1['frames']) & set(obj2['frames']))

            if common_frames:
                distance_list = []

                for frame in common_frames:
                    if DATASET == 'KITTI':
                        frame_str = f"{frame:06d}"
                    else:
                        frame_str = f"{frame:05d}"

                    # Get 3D coordinates of objects in this frame
                    if frame_str in obj1['track_data'] and frame_str in obj2['track_data']:
                        obj1_coords = obj1['track_data'][frame_str].get('3d_cord')
                        obj2_coords = obj2['track_data'][frame_str].get('3d_cord')

                        if obj1_coords and obj2_coords:
                            # Calculate Euclidean distance
                            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(obj1_coords, obj2_coords)))
                            distance_list.append(distance)

                if distance_list:
                    # Store distances for both (obj1, obj2) and (obj2, obj1)
                    distances[(obj1_id, obj2_id)] = distance_list
                    distances[(obj2_id, obj1_id)] = distance_list

    return distances


def build_kb_for_window(track_data, window_start, window_size=10, rules_file=None):
    """
    Build Knowledge Base for a specific window

    Args:
        track_data: Dictionary of track data by class
        window_start: Starting frame number
        window_size: Number of frames in window
        rules_file: Path to file containing implication rules

    Returns:
        Knowledge Base object and list of predicates
    """
    # Get frames in this window
    window_frames = get_frames_in_window(track_data, window_start, window_size)
    # print(window_frames)
    if not window_frames:
        print(f"No frames found in window starting at {window_start}")
        return None, []

    # print(f"Building KB for window {window_start} to {window_start + window_size - 1}")

    # Get objects in this window
    objects = get_objects_in_window(track_data, window_frames)
    # print(f"Found {len(objects)} objects in window")

    # Calculate distances between objects
    distances = calculate_distances(objects, window_frames)

    # Initialize Knowledge Base with implications
    KB = None
    if rules_file and os.path.exists(rules_file):
        with open(rules_file, 'r') as f:
            rules = [line.strip() for line in f.readlines() if line.strip()]
        clauses = [aima.utils.expr(rule) for rule in rules]
        KB = aima.logic.FolKB(clauses)
    else:
        KB = aima.logic.FolKB([])

    # List to store all predicates added to KB
    all_predicates = []

    # Add predicates for each object
    for obj_id, obj_info in objects.items():
        class_name = obj_info['class']
        frames = obj_info['frames']
        all_frames = obj_info['all_frames']
        track_data = obj_info['track_data']

        # Skip non-valid classes
        if not class_name or class_name == 'unlabeled':
            continue

        # 1. Add class type predicate
        class_predicate = f"{class_name.capitalize()}({obj_id})"
        KB.tell(aima.utils.expr(class_predicate))
        all_predicates.append(class_predicate)

        # 2. Check appearance/disappearance in window
        first_appearance = min(all_frames)
        appears_in_window = False

        if window_start <= first_appearance <= window_start + window_size - 1:
            # Only check the 5 frames immediately before first appearance
            frames_before = range(max(0, first_appearance - 5), first_appearance)
            appears_in_window = not any(f in all_frames for f in frames_before)

        # Check if object disappears within the window
        last_appearance = max(all_frames)
        disappears_in_window = False

        # print(first_appearance, frames, last_appearance)

        if window_start <= last_appearance <= window_start + window_size - 1:
            # Only check the 5 frames immediately after last appearance
            frames_after = range(last_appearance + 1, last_appearance + 6)
            disappears_in_window = not any(f in all_frames for f in frames_after)

        if appears_in_window:
            KB.tell(aima.utils.expr(f"Appear({obj_id})"))
            all_predicates.append(f"Appear({obj_id})")

        if disappears_in_window:
            KB.tell(aima.utils.expr(f"Disappear({obj_id})"))
            all_predicates.append(f"Disappear({obj_id})")

        # 3. Determine initial and last positions
        if frames:
            if DATASET == 'KITTI':
                first_frame = str(frames[0]).zfill(6)
                last_frame = str(frames[-1]).zfill(6)
            else:
                first_frame = str(frames[0]).zfill(5)
                last_frame = str(frames[-1]).zfill(5)

            # Get 3D coordinates
            first_coords = track_data.get(first_frame, {}).get('3d_cord')
            last_coords = track_data.get(last_frame, {}).get('3d_cord')

            cent_first = track_data.get(first_frame, {}).get('centroid')
            cent_last = track_data.get(last_frame, {}).get('centroid')

            # first_center = [
            #     (b_box_first[0] + b_box_first[2]//2)
            #     (b_box_first[0] + b_box_first[2]//2),

            if first_coords and last_coords:
                # Determine positions
                if DATASET == 'KITTI':
                    max_x = 1242  # image width
                else:
                    max_x = 480

                first_x, first_z = cent_first[0], first_coords[2]
                last_x, last_z = cent_last[0], last_coords[2]

                first_pos_x, first_pos_z = get_position_category(first_x, first_z, max_x)
                last_pos_x, last_pos_z = get_position_category(last_x, last_z, max_x)

                initial_location = f"InitialLocation({obj_id}, {first_pos_z.capitalize()}{first_pos_x.capitalize()})"
                last_location = f"LastLocation({obj_id}, {last_pos_z.capitalize()}{last_pos_x.capitalize()})"

                KB.tell(aima.utils.expr(initial_location))
                KB.tell(aima.utils.expr(last_location))

                initial_location = f"InitialLocation({obj_id}, {first_pos_x.capitalize()})"
                last_location = f"LastLocation({obj_id}, {last_pos_x.capitalize()})"

                KB.tell(aima.utils.expr(initial_location))
                KB.tell(aima.utils.expr(last_location))

                all_predicates.append(initial_location)
                all_predicates.append(last_location)

        # 4. For moving objects (vehicles, pedestrians), determine speed and acceleration
        if class_name in ['vehicles', 'pedestrian']:
            # Get displacement values for all frames
            displacement_values = []

            for frame in frames:
                if DATASET == 'KITTI':
                    frame_str = str(frame).zfill(6)
                else:
                    frame_str = str(frame).zfill(5)
                if frame_str in track_data:
                    displacement = track_data[frame_str].get('displacement')
                    if displacement is not None:
                        displacement_values.append(displacement)

            # Process displacement data to determine movement
            if displacement_values:
                # Filter outliers
                filtered_displacements = filter_outliers(displacement_values)

                # print(f"{obj_id}")
                # print(displacement_values)
                # print(filtered_displacements)
                # print(np.mean(filtered_displacements))
                # print("*"*40)

                # Calculate average displacement
                avg_displacement = np.mean(filtered_displacements)

                # Determine if object is moving
                threshold = SPEED_THRESHOLD_VEHICLE if class_name == 'vehicles' else SPEED_THRESHOLD_PEDESTRIAN
                is_moving = avg_displacement > threshold

                # Set Moving/NotMoving predicates
                if is_moving:
                    KB.tell(aima.utils.expr(f"Moving({obj_id})"))
                    all_predicates.append(f"Moving({obj_id})")

                    # For moving objects, determine if speed is changing
                    slope = calculate_slope(filtered_displacements)

                    # print(f"{obj_id}")
                    # print(displacement_values)
                    # print(filtered_displacements)
                    # print(slope)
                    # print("*" * 40)

                    if slope > VEL_SLOPE_THRESHOLD:
                        KB.tell(aima.utils.expr(f"SpeedUp({obj_id})"))
                        all_predicates.append(f"SpeedUp({obj_id})")
                    else:
                        KB.tell(aima.utils.expr(f"NotSpeedUp({obj_id})"))
                        all_predicates.append(f"NotSpeedUp({obj_id})")

                    if slope < -VEL_SLOPE_THRESHOLD:
                        KB.tell(aima.utils.expr(f"SpeedDown({obj_id})"))
                        all_predicates.append(f"SpeedDown({obj_id})")
                    else:
                        KB.tell(aima.utils.expr(f"NotSpeedDown({obj_id})"))
                        all_predicates.append(f"NotSpeedDown({obj_id})")
                else:
                    KB.tell(aima.utils.expr(f"NotMoving({obj_id})"))
                    all_predicates.append(f"NotMoving({obj_id})")

                    # Non-moving objects have no speed change
                    KB.tell(aima.utils.expr(f"NotSpeedUp({obj_id})"))
                    KB.tell(aima.utils.expr(f"NotSpeedDown({obj_id})"))
                    all_predicates.append(f"NotSpeedUp({obj_id})")
                    all_predicates.append(f"NotSpeedDown({obj_id})")

        # 5. For vehicles, check specific vehicle type
        if class_name == 'vehicles':
            # Try to get vehicle type
            vehicle_type = None

            # Look for type information in the first frame
            if frames:
                if DATASET == 'KITTI':
                    first_frame = str(frames[0]).zfill(6)
                else:
                    first_frame = str(frames[0]).zfill(5)

                if 'type' in track_data.get(first_frame, {}):
                    vehicle_type = track_data[first_frame]['type']

                # If model predictions are available, use those
                if 'pred_type' in track_data.get(first_frame, {}):
                    pred_type = track_data[first_frame]['pred_type']
                    if pred_type and isinstance(pred_type, dict) and len(pred_type) > 0:
                        # Get most confident vehicle type
                        vehicle_type = max(pred_type.items(), key=lambda x: x[1])[0]

            # Map vehicle type to FOL predicate if available
            # if vehicle_type and vehicle_type.lower() in VEHICLE_TYPE_MAPPING:
            #     predicate_type = VEHICLE_TYPE_MAPPING[vehicle_type.lower()]
            #     KB.tell(aima.utils.expr(f"{predicate_type}({obj_id})"))
            #     all_predicates.append(f"{predicate_type}({obj_id})")

    # 6. Process distances between objects
    for (obj1_id, obj2_id), distance_list in distances.items():
        # Skip if either object is no longer in objects dict (might have been filtered)
        if obj1_id not in objects or obj2_id not in objects:
            continue

        # Filter outliers
        filtered_distances = filter_outliers(distance_list)

        # Calculate slope to determine if distance is increasing/decreasing
        slope = calculate_slope(filtered_distances)

        # if 'Vehicle' in obj1_id and 'Vehicle' in obj2_id:
        #     print(f"{obj1_id}, {obj2_id}")
        #     print(distance_list)
        #     print(filtered_distances)
        #     print(slope)
        #     print("*"*40)

        # Check if objects are very close
        if np.min(filtered_distances) < VERY_CLOSE_DISTANCE:
            KB.tell(aima.utils.expr(f"TooClose({obj1_id}, {obj2_id})"))
            all_predicates.append(f"TooClose({obj1_id}, {obj2_id})")

        # Check if distance starts or ends at zero
        if filtered_distances and filtered_distances[0] < ZERO_DISTANCE_THRESHOLD:
            KB.tell(aima.utils.expr(f"DistanceZeroStart({obj1_id}, {obj2_id})"))
            all_predicates.append(f"DistanceZeroStart({obj1_id}, {obj2_id})")

        if filtered_distances and filtered_distances[-1] < ZERO_DISTANCE_THRESHOLD:
            KB.tell(aima.utils.expr(f"DistanceZeroEnd({obj1_id}, {obj2_id})"))
            all_predicates.append(f"DistanceZeroEnd({obj1_id}, {obj2_id})")

        # Determine if distance is increasing or decreasing
        if slope > DIST_SLOPE_THRESHOLD:
            KB.tell(aima.utils.expr(f"DistanceIncrease({obj1_id}, {obj2_id})"))
            all_predicates.append(f"DistanceIncrease({obj1_id}, {obj2_id})")
        else:
            KB.tell(aima.utils.expr(f"NotDistanceIncrease({obj1_id}, {obj2_id})"))
            all_predicates.append(f"NotDistanceIncrease({obj1_id}, {obj2_id})")

        if slope < -DIST_SLOPE_THRESHOLD:
            KB.tell(aima.utils.expr(f"DistanceDecrease({obj1_id}, {obj2_id})"))
            all_predicates.append(f"DistanceDecrease({obj1_id}, {obj2_id})")
        else:
            KB.tell(aima.utils.expr(f"NotDistanceDecrease({obj1_id}, {obj2_id})"))
            all_predicates.append(f"NotDistanceDecrease({obj1_id}, {obj2_id})")

    # 7. Determine "On" relationships
    for obj1_id, obj1_info in objects.items():
        # Only pedestrians can be "on" something
        if obj1_info['class'] != 'pedestrian':
            continue

        for obj2_id, obj2_info in objects.items():
            # Skip self
            if obj1_id == obj2_id:
                continue

            # Only consider vehicles, roads, sidewalks
            if obj2_info['class'] not in ['vehicles', 'road', 'sidewalk']:
                continue

            # Find common frames
            common_frames = set(obj1_info['frames']) & set(obj2_info['frames'])

            if common_frames:
                # Check if obj1 (pedestrian) is on obj2
                on_count = 0

                for frame in common_frames:
                    if DATASET == 'KITTI':
                        frame_str = str(frame).zfill(6)
                    else:
                        frame_str = str(frame).zfill(5)

                    # Get bounding boxes
                    if frame_str in obj1_info['track_data'] and frame_str in obj2_info['track_data']:
                        ped_bbox = obj1_info['track_data'][frame_str].get('b_box')
                        obj_bbox = obj2_info['track_data'][frame_str].get('b_box')

                        if ped_bbox and obj_bbox:
                            # Check if pedestrian is within obj2's bounding box
                            px, py, pw, ph = ped_bbox
                            ox, oy, ow, oh = obj_bbox

                            # Simple check: pedestrian bottom center is inside obj2
                            ped_bottom_x = px + pw // 2
                            ped_bottom_y = py + ph

                            if (ox <= ped_bottom_x <= ox + ow) and (oy <= ped_bottom_y <= oy + oh):
                                on_count += 1

                # If pedestrian is on obj2 in at least 50% of common frames
                if on_count >= len(common_frames) * 0.5:
                    KB.tell(aima.utils.expr(f"On({obj1_id}, {obj2_id})"))
                    all_predicates.append(f"On({obj1_id}, {obj2_id})")

    return KB, all_predicates


def save_kb(KB, predicates, output_file):
    """
    Save Knowledge Base and predicates to file

    Args:
        KB: Knowledge Base object
        predicates: List of predicates in the KB
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        # f.write("# First-Order Logic Knowledge Base\n\n")
        #
        # # Write all predicates
        # f.write("# Base predicates:\n")
        # for predicate in predicates:
        #     f.write(f"{predicate}\n")
        #
        # f.write("\n# Knowledge Base clauses:\n")
        for clause in KB.clauses:
            f.write(f"{clause}\n")


def process_video(video_name, tracker_dir, output_dir, rules_file=None, window_size=10):
    """
    Process a video and generate knowledge bases for each window

    Args:
        video_name: Name of the video
        tracker_dir: Directory containing tracker data
        output_dir: Directory to save output files
        rules_file: Path to file containing implication rules
        window_size: Window size in frames
    """
    print(f"Processing video: {video_name}")

    # Load tracker data
    track_data = load_tracker_data(tracker_dir, video_name)

    if not track_data:
        print(f"No tracker data found for video {video_name}")
        return

    # Create output directory
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Find all frames in the video
    all_frames = set()
    for class_name, class_data in track_data.items():
        for track_id, track_info in class_data.items():
            for frame_num in track_info.keys():
                all_frames.add(int(frame_num))

    all_frames = natsorted(list(all_frames))

    if not all_frames:
        print(f"No frames found for video {video_name}")
        return

    # Process each window
    for window_start in range(all_frames[0], all_frames[-1], window_size):
        # Build KB for this window
        KB, predicates = build_kb_for_window(track_data, window_start, window_size, rules_file)

        if KB and predicates:
            # Save KB to file
            if DATASET == 'KITTI':
                output_file = os.path.join(
                    video_output_dir,
                    f"kb_window_{window_start:06d}_{window_start + window_size - 1:06d}.txt"
                )
            else:
                output_file = os.path.join(
                    video_output_dir,
                    f"kb_window_{window_start:05d}_{window_start + window_size - 1:05d}.txt"
                )

            save_kb(KB, predicates, output_file)
            # print(f"Saved KB for window {window_start} to {window_start + window_size - 1}")


def process_video_parallel(args):
    """Wrapper function for process_video to use with multiprocessing"""
    video_name, tracker_dir, output_dir, rules_file, window_size = args
    process_video(video_name, tracker_dir, output_dir, rules_file, window_size)
    return f"Completed processing {video_name}"


def main():
    global DATASET

    parser = argparse.ArgumentParser(description="Convert tracker data to First-Order Logic Knowledge Base")
    parser.add_argument("--tracker_dir", required=True, help="Directory containing tracker data")
    parser.add_argument("--output_dir", required=True, help="Directory to save output files")
    parser.add_argument("--rules_file", help="Path to file containing implication rules")
    parser.add_argument("--dataset", help="Name of the Dataset")
    parser.add_argument("--window_size", type=int, default=10, help="Window size in frames")
    parser.add_argument("--num_processes", type=int, default=8,
                        help="Number of processes to use (default: number of CPU cores)")

    args = parser.parse_args()

    DATASET = args.dataset.upper()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    video_list = [vid for vid in os.listdir(args.tracker_dir) if os.path.isdir(os.path.join(args.tracker_dir, vid))]
    video_list = natsorted(video_list)  # [10:]  # [16:]  # [15:16]  # [:1]  # [1:2]
    # print(video_list)

    # for video_name in tqdm(video_list):
    #     process_video(video_name, args.tracker_dir, args.output_dir, args.rules_file, args.window_size)

    num_processes = args.num_processes or multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for parallel processing")

    # Prepare arguments for each video
    process_args = []
    for video_name in video_list:
        process_args.append((video_name, args.tracker_dir, args.output_dir, args.rules_file,
                             args.window_size))

    # Process videos in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_video_parallel, process_args)

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()

# python final_KB_gen.py - -tracker_dir / home / ibk5106 / projects / projects / LogicRAG / track_out - -output_dir / home / ibk5106 / projects / projects / LogicRAG / kb_out - -rules_file / home / ibk5106 / projects / projects / LogicRAG / tools / kb_framework / rules / all_rules.txt
