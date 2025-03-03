#!/usr/bin/env python3
# Tracker that combines panoptic segmentation, optical flow, and depth for object tracking
import os
import cv2
import numpy as np
import glob
import pickle
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from tqdm import tqdm
from natsort import natsorted
from PIL import Image
import torch
import multiprocessing
from functools import partial

DATASET = 'CARLA'

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


# Define label mapping from Cityscapes to our custom mapping
def map_labels_to_custom(label_id):
    # Map Cityscapes trainIds to our custom labels
    mapping = {
        # Void/Ignore classes -> unlabeled (0)
        255: 0,
        # Construction
        2: 1,  # building
        3: 11,  # wall
        4: 2,  # fence
        # Human
        11: 4,  # person/pedestrian
        12: 4,  # rider -> pedestrian
        # Vehicle
        13: 10,  # car -> vehicles
        14: 10,  # truck -> vehicles
        15: 10,  # bus -> vehicles
        16: 10,  # train -> vehicles
        17: 10,  # motorcycle -> vehicles
        18: 10,  # bicycle -> vehicles
        # Flat
        0: 7,  # road
        1: 8,  # sidewalk
        # Object
        5: 5,  # pole
        6: 18,  # traffic light
        7: 12,  # traffic sign
        # Nature
        8: 9,  # vegetation
        9: 22,  # terrain
        # Sky
        10: 13,  # sky
    }

    return mapping.get(label_id, 3)  # Default to 'other' (3) if not found


class InstanceTracker:
    def __init__(self, max_lost_frames=5, iou_threshold=0.1, flow_weight=0.5, min_obj_size=10,
                 type_model_path=None, color_model_path=None):
        self.max_lost_frames = max_lost_frames
        self.iou_threshold = iou_threshold
        self.flow_weight = flow_weight
        self.iou_weight = 1.0 - flow_weight
        self.next_track_id = 1
        self.tracks = {}  # {track_id: {'class_id': class_id, 'frames': {frame_idx: data}}}
        self.min_obj_size = min_obj_size

        self.next_track_id = 0

        # Initialize model loader flags
        self.models_loaded = False

        # Initialize models if paths are provided
        if type_model_path is not None and color_model_path is not None:
            self.initialize_ml_models(type_model_path, color_model_path)

    def get_vehicle_type(self, class_mask, instance_mask):
        """Determine the specific type of vehicle from the class mask"""
        # Get all class IDs in this vehicle instance
        instance_classes = class_mask[instance_mask]
        if len(instance_classes) == 0:
            return "car"  # Default if no classes found

        # Map from trainId to type
        type_mapping = {
            13: "Car",
            14: "Truck",
            15: "Bus",
            16: "Train",
            17: "Motorcycle",
            18: "Bicycle",
        }

        # Count occurrences of each class
        class_counts = np.bincount(instance_classes)
        # Find the most common class (excluding background)
        # if len(class_counts) > 0:
        most_common = np.argmax(class_counts)
        # print(most_common, type_mapping.get(most_common))
        return type_mapping.get(most_common, "car")

        # return "car"  # Default

    def detect_vehicle_color(self, rgb_image, instance_mask):
        """Detect the dominant color of a vehicle"""
        # Common vehicle colors and their RGB values
        colors = {
            "Black": ([0, 0, 0], [50, 50, 50]),
            "White": ([200, 200, 200], [255, 255, 255]),
            "Gray": ([100, 100, 100], [180, 180, 180]),
            "Red": ([150, 0, 0], [255, 80, 80]),
            "Blue": ([0, 0, 100], [80, 80, 255]),
            "Green": ([0, 100, 0], [80, 255, 80]),
            "Yellow": ([200, 200, 0], [255, 255, 100]),
            "Orange": ([200, 100, 0], [255, 180, 50]),
            "Brown": ([100, 50, 0], [150, 100, 50]),
            "Silver": ([180, 180, 180], [210, 210, 210])
        }

        # Apply mask to extract vehicle pixels
        if rgb_image is None or not np.any(instance_mask):
            return "unknown"

        masked_vehicle = cv2.bitwise_and(rgb_image, rgb_image, mask=instance_mask.astype(np.uint8))
        vehicle_pixels = masked_vehicle[instance_mask]

        if len(vehicle_pixels) == 0:
            return "unknown"

        # Calculate average color
        avg_color = np.mean(vehicle_pixels, axis=0)

        # Find the closest color
        min_dist = float('inf')
        detected_color = "unknown"

        for color_name, (lower, upper) in colors.items():
            lower = np.array(lower)
            upper = np.array(upper)

            if np.all(avg_color >= lower) and np.all(avg_color <= upper):
                # Color is within range
                return color_name

            # Calculate distance to the center of the color range
            color_center = (np.array(lower) + np.array(upper)) / 2
            dist = np.sum((avg_color - color_center) ** 2)

            if dist < min_dist:
                min_dist = dist
                detected_color = color_name

        return detected_color

    def initialize_ml_models(self, type_model_path, color_model_path):
        """Initialize and load the type and color prediction models"""
        try:
            import torch
            import torchvision.transforms as transforms
            from torchvision import models
            import torch.nn as nn

            # Define class lists
            self.vehicle_types = ['Cab', 'Convertible', 'Coupe', 'Hatchback', 'Minivan', 'Other', 'SUV', 'Sedan', 'Van',
                                  'Wagon']
            self.vehicle_colors = ['Black', 'Blue', 'Brown', 'Cyan', 'Green', 'Grey', 'Orange', 'Red', 'Violet',
                                   'White', 'Yellow', 'Other']

            # Define transform for images
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize and load type model
            self.type_model = self.initialize_model(len(self.vehicle_types))
            checkpoint = torch.load(type_model_path, map_location=self.device)
            self.type_model.load_state_dict(checkpoint['model_state_dict'])
            self.type_model.to(self.device)
            self.type_model.eval()

            # Initialize and load color model
            self.color_model = self.initialize_model(len(self.vehicle_colors))
            checkpoint = torch.load(color_model_path, map_location=self.device)
            self.color_model.load_state_dict(checkpoint['model_state_dict'])
            self.color_model.to(self.device)
            self.color_model.eval()

            self.models_loaded = True
            print("Successfully loaded models for vehicle type and color prediction")

        except Exception as e:
            self.models_loaded = False
            print(f"Error loading models: {e}")

    def initialize_model(self, num_classes):
        """Initialize EfficientNet model with the specified number of classes"""
        try:
            import torch.nn as nn
            from torchvision import models

            # Load a pre-trained EfficientNet model
            model = models.efficientnet_b0(weights='IMAGENET1K_V1')

            # Freeze early layers
            for param in list(model.parameters())[:-20]:
                param.requires_grad = False

            # Replace the final fully connected layer
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, num_classes)

            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None

    def predict_with_models(self, rgb_image, instance_mask):
        """Predict vehicle type and color using the loaded models"""
        if not self.models_loaded or rgb_image is None or not np.any(instance_mask):
            return {"unknown": 1.0}, {"unknown": 1.0}

        try:
            import torch
            from PIL import Image
            import io

            # Create a cropped image of just the vehicle
            y_indices, x_indices = np.where(instance_mask)
            if len(y_indices) == 0:
                return {"unknown": 1.0}, {"unknown": 1.0}

            # Get bounding box of the mask
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)

            # Create a copy of the cropped vehicle
            vehicle_crop = rgb_image[y_min:y_max, x_min:x_max].copy()

            # Further refine the crop by applying the mask
            mask_crop = instance_mask[y_min:y_max, x_min:x_max]
            vehicle_crop[~mask_crop] = [0, 0, 0]  # Set background to black

            # Convert to PIL Image
            vehicle_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_RGB2BGR)
            pil_image = Image.fromarray(vehicle_crop)

            # Apply transformations and add batch dimension
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

            # Get predictions
            # Get predictions
            with torch.no_grad():
                # Type prediction
                type_outputs = self.type_model(input_tensor)
                type_probs = torch.nn.functional.softmax(type_outputs, dim=1)[0]

                # Color prediction
                color_outputs = self.color_model(input_tensor)
                color_probs = torch.nn.functional.softmax(color_outputs, dim=1)[0]

            # Get top 3 predictions for type
            type_values, type_indices = torch.topk(type_probs, min(3, len(self.vehicle_types)))

            # Get top 3 predictions for color
            color_values, color_indices = torch.topk(color_probs, min(3, len(self.vehicle_colors)))

            # Create dictionaries with predictions
            type_dict = {self.vehicle_types[idx.item()]: val.item() for val, idx in zip(type_values, type_indices)}
            color_dict = {self.vehicle_colors[idx.item()]: val.item() for val, idx in zip(color_values, color_indices)}

            return type_dict, color_dict

        except Exception as e:
            print(f"Error in model prediction: {e}")
            return {"unknown": 1.0}, {"unknown": 1.0}

    def load_label_mask(self, path):
        """Load panoptic label mask with class_id in first channel and instance_id in the other channels"""
        mask = np.array(Image.open(path))

        class_mask = mask[:, :, 0].astype(np.int32)
        instance_id = (mask[:, :, 1].astype(np.int32) << 8) | mask[:, :, 2].astype(np.int32)

        # print(np.unique(class_mask), np.unique(instance_id))

        return class_mask, instance_id

    def load_depth(self, path, img_shape):
        """
        Load depth map and pad if smaller than image_shape

        Args:
            path: Path to depth map file
            img_shape: (height, width) of the target image

        Returns:
            Padded depth map or None if loading fails
        """
        target_h, target_w = img_shape

        # Load depth map
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        # Convert depth from uint16 to float meters (KITTI format)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / SCALE

        # Check if depth map needs padding
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

    def load_flow(self, path):
        """Load optical flow data"""
        if os.path.exists(path):
            try:
                flow_data = np.load(path)
                return flow_data
            except Exception as e:
                print(f"Error loading flow data from {path}: {e}")
        return None

    # def extract_instances(self, class_mask, instance_id, frame_idx, rgb_image=None):
    #     """Extract instance information from masks"""
    #     instances = []
    #     unique_instances = np.unique(instance_id)
    #
    #     for inst_id in unique_instances:
    #         if inst_id == 0:  # Skip background
    #             continue
    #
    #         # Create binary mask for this instance
    #         inst_mask = (instance_id == inst_id)
    #
    #         # Get the class ID for this instance (use the most common class in the instance)
    #         class_ids = class_mask[inst_mask]
    #         if len(class_ids) == 0:
    #             continue
    #
    #         most_common_class = np.bincount(class_ids).argmax()
    #         mapped_class = map_labels_to_custom(most_common_class)
    #
    #         # print(np.unique(class_ids), most_common_class,  mapped_class)
    #
    #         # print(inst_mask)
    #
    #         # Skip very small objects
    #         contours, _ = cv2.findContours(inst_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         if not contours:
    #             continue
    #
    #         largest_contour = max(contours, key=cv2.contourArea)
    #         area = cv2.contourArea(largest_contour)
    #         x, y, w, h = cv2.boundingRect(largest_contour)
    #
    #         # Skip if both width and height are too small
    #         if w < self.min_obj_size and h < self.min_obj_size:
    #             continue
    #
    #         # Calculate centroid
    #         M = cv2.moments(largest_contour)
    #         if M["m00"] > 0:
    #             cx = int(M["m10"] / M["m00"])
    #             cy = int(M["m01"] / M["m00"])
    #         else:
    #             cx, cy = x + w // 2, y + h // 2
    #
    #         # Store instance info
    #         if mapped_class == 10:  # vehicles class
    #             # Get the specific vehicle type
    #             vehicle_type = self.get_vehicle_type(class_mask, inst_mask)
    #             # Get the vehicle color if RGB image is available
    #             vehicle_color = self.detect_vehicle_color(rgb_image, inst_mask) if rgb_image is not None else "unknown"
    #
    #             vehicle_type = self.get_vehicle_type(class_mask, inst_mask)
    #             # Get the vehicle color using the current method
    #             vehicle_color = self.detect_vehicle_color(rgb_image, inst_mask) if rgb_image is not None else "unknown"
    #
    #             # Get model predictions if models are loaded
    #             pred_type, pred_color = {}, {}
    #             if rgb_image is not None and self.models_loaded:
    #                 pred_type, pred_color = self.predict_with_models(rgb_image, inst_mask)
    #
    #             # print(pred_type, pred_color)
    #
    #             # Add to instance info
    #             instances.append({
    #                 'class_id': mapped_class,
    #                 'instance_id': inst_id,
    #                 'mask': inst_mask,
    #                 'bbox': (x, y, w, h),
    #                 'centroid': (cx, cy),
    #                 'area': area,
    #                 'frame_idx': frame_idx,
    #                 'type': vehicle_type,
    #                 'color': vehicle_color,
    #                 'pred_type': pred_type,
    #                 'pred_color': pred_color
    #             })
    #         else:
    #             instances.append({
    #                 'class_id': mapped_class,
    #                 'instance_id': inst_id,
    #                 'mask': inst_mask,
    #                 'bbox': (x, y, w, h),
    #                 'centroid': (cx, cy),
    #                 'area': area,
    #                 'frame_idx': frame_idx,
    #                 'type': 'NA',
    #                 'color': 'NA',
    #                 'pred_type': 'NA',
    #                 'pred_color': 'NA'
    #             })
    #
    #     # print(instances)
    #
    #     return instances

    def extract_instances(self, class_mask, instance_id, frame_idx, rgb_image=None):
        """Extract instance information from masks"""
        instances = []

        # First, handle vehicles and pedestrians as before using instance_id
        unique_instances = np.unique(instance_id)
        for inst_id in unique_instances:
            if inst_id == 0:  # Skip background
                continue

            # Create binary mask for this instance
            inst_mask = (instance_id == inst_id)

            # Get the class ID for this instance (use the most common class in the instance)
            class_ids = class_mask[inst_mask]
            if len(class_ids) == 0:
                continue

            most_common_class = np.bincount(class_ids).argmax()
            mapped_class = map_labels_to_custom(most_common_class)

            # Only process vehicles and pedestrians here (classes 10 and 4)
            if mapped_class in [10, 4]:
                # Skip very small objects
                contours, _ = cv2.findContours(inst_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Skip if both width and height are too small
                if w < self.min_obj_size and h < self.min_obj_size:
                    continue

                # Calculate centroid
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                # Store instance info
                if mapped_class == 10:  # vehicles class
                    # Get the specific vehicle type
                    vehicle_type = self.get_vehicle_type(class_mask, inst_mask)
                    # Get the vehicle color if RGB image is available
                    vehicle_color = self.detect_vehicle_color(rgb_image,
                                                              inst_mask) if rgb_image is not None else "unknown"

                    # Get model predictions if models are loaded
                    pred_type, pred_color = {}, {}
                    if rgb_image is not None and self.models_loaded:
                        pred_type, pred_color = self.predict_with_models(rgb_image, inst_mask)

                    instances.append({
                        'class_id': mapped_class,
                        'instance_id': inst_id,
                        'mask': inst_mask,
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'area': area,
                        'frame_idx': frame_idx,
                        'type': vehicle_type,
                        'color': vehicle_color,
                        'pred_type': pred_type,
                        'pred_color': pred_color
                    })
                else:  # pedestrian class
                    instances.append({
                        'class_id': mapped_class,
                        'instance_id': inst_id,
                        'mask': inst_mask,
                        'bbox': (x, y, w, h),
                        'centroid': (cx, cy),
                        'area': area,
                        'frame_idx': frame_idx,
                        'type': 'NA',
                        'color': 'NA',
                        'pred_type': 'NA',
                        'pred_color': 'NA'
                    })

        # Now, handle other classes using connected components
        # Create a dictionary to track instance counts per class
        class_instance_counters = {}

        # Create a mask for all classes except vehicles and pedestrians
        other_classes_mask = ~np.isin(class_mask, [10, 4])

        # Get unique class IDs in the mask
        unique_other_classes = np.unique(class_mask[other_classes_mask])

        # Process each class separately
        for class_id in unique_other_classes:
            if class_id == 0:  # Skip background
                continue

            mapped_class = map_labels_to_custom(class_id)

            # Skip if mapped to vehicle or pedestrian (already handled)
            if mapped_class in [10, 4]:
                continue

            # Initialize instance counter for this class if not exists
            if mapped_class not in class_instance_counters:
                class_instance_counters[mapped_class] = 0

            # Create binary mask for this class
            class_binary_mask = (class_mask == class_id)

            # Find connected components
            contours, _ = cv2.findContours(class_binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

            # Process each connected component as a separate instance
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)

                # Skip small objects
                if w < self.min_obj_size and h < self.min_obj_size:
                    continue

                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2

                # Create mask for this contour
                inst_mask = np.zeros_like(class_binary_mask).astype(np.uint8)
                cv2.drawContours(inst_mask, [contour], 0, 1, thickness=cv2.FILLED)

                # Increment instance counter for this class
                class_instance_counters[mapped_class] += 1

                # Create a unique instance ID: 1000000 + class_id*10000 + counter
                # This ensures no overlap with existing instance IDs
                new_inst_id = 1000000 + mapped_class * 10000 + class_instance_counters[mapped_class]

                instances.append({
                    'class_id': mapped_class,
                    'instance_id': new_inst_id,
                    'mask': inst_mask,
                    'bbox': (x, y, w, h),
                    'centroid': (cx, cy),
                    'area': area,
                    'frame_idx': frame_idx,
                    'type': 'NA',
                    'color': 'NA',
                    'pred_type': 'NA',
                    'pred_color': 'NA'
                })

        return instances

    def compute_depth_info(self, instance, depth_map):
        """Compute 3D coordinates from depth map"""
        if depth_map is None:
            return None

        cx, cy = instance['centroid']

        # Ensure coordinates are within bounds
        if not (0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]):
            return None

        # Get depth at centroid
        z = depth_map[cy, cx]

        # Convert to 3D coordinates
        x = (cx - CX_DEPTH) * z / FX_DEPTH
        y = (cy - CY_DEPTH) * z / FY_DEPTH

        return [x, y, z, cx, cy]

    def compute_iou(self, mask1, mask2):
        """Compute IoU between two binary masks"""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    def predict_mask_from_flow(self, prev_mask, flow_data, class_id):
        """flow prediction using KD-tree"""
        if flow_data is None or len(flow_data) == 0:
            return None

        # Get points in the previous mask
        y_indices, x_indices = np.where(prev_mask)
        if len(y_indices) == 0:
            return None

        h, w = prev_mask.shape

        # Get the flow source coordinates and corresponding target positions
        flow_sources = flow_data[:, :2]  # Source coordinates (cols 0-1)
        flow_targets = flow_data[:, 4:6]  # Target coordinates (cols 4-5)

        # Build a spatial index for fast nearest neighbor lookup
        from scipy.spatial import cKDTree
        kdtree = cKDTree(flow_sources)

        # Query points (from the mask)
        query_points = np.column_stack((x_indices, y_indices))

        # Find nearest neighbors for all query points at once
        _, indices = kdtree.query(query_points, k=1)

        # Get the corresponding target positions
        new_positions = flow_targets[indices]

        # Round to integers and clip to image boundaries
        new_positions = np.round(new_positions).astype(int)
        new_positions[:, 0] = np.clip(new_positions[:, 0], 0, w - 1)
        new_positions[:, 1] = np.clip(new_positions[:, 1], 0, h - 1)

        # Create predicted mask
        predicted_mask = np.zeros((h, w), dtype=bool)
        predicted_mask[new_positions[:, 1], new_positions[:, 0]] = True

        return predicted_mask, new_positions

    def compute_flow_cost(self, prev_instance, curr_instance, flow_data, class_id):
        """Compute cost based on flow prediction using point-based accuracy instead of IoU"""
        if flow_data is None:
            print("flow none")
            return 1.0  # Maximum cost if no flow data

        # Predict where the previous mask would move according to flow
        pred_mask, _ = self.predict_mask_from_flow(prev_instance['mask'], flow_data, class_id)
        if pred_mask is None:
            print("pred none")
            return 1.0

        # Count how many predicted points fall within the current mask
        overlap = np.logical_and(pred_mask, curr_instance['mask']).sum()

        # Total number of predicted points
        total_predicted = np.sum(pred_mask)

        if total_predicted == 0:
            return 1.0  # No prediction, maximum cost

        accuracy = overlap / total_predicted

        return 1.0 - accuracy

    def compute_distance_cost(self, centroid1, centroid2, max_dist=100.0):
        """Compute normalized distance between centroids"""
        cx1, cy1 = centroid1
        cx2, cy2 = centroid2
        dist = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)
        return min(dist / max_dist, 1.0)  # Normalize to [0, 1]

    def assign_tracks(self, prev_instances, curr_instances, flow_data):
        """Assign current instances to existing tracks using Hungarian algorithm, one class at a time"""
        if not prev_instances or not curr_instances:
            return [], []

        # Group instances by class
        prev_by_class = {}
        curr_by_class = {}

        for i, inst in enumerate(prev_instances):
            class_id = inst['class_id']
            if class_id not in prev_by_class:
                prev_by_class[class_id] = []
            prev_by_class[class_id].append((i, inst))

        for j, inst in enumerate(curr_instances):
            class_id = inst['class_id']
            if class_id not in curr_by_class:
                curr_by_class[class_id] = []
            curr_by_class[class_id].append((j, inst))

        # Process each class separately
        all_assignments = []
        all_unassigned_curr = list(range(len(curr_instances)))

        for class_id, prev_class_instances in prev_by_class.items():
            if class_id not in curr_by_class:
                continue

            curr_class_instances = curr_by_class[class_id]

            # Get original indices
            prev_indices = [i for i, _ in prev_class_instances]
            curr_indices = [j for j, _ in curr_class_instances]

            # Build cost matrix for this class
            cost_matrix = np.zeros((len(prev_class_instances), len(curr_class_instances)))

            for i, (_, prev_inst) in enumerate(prev_class_instances):
                for j, (_, curr_inst) in enumerate(curr_class_instances):
                    # Calculate costs
                    iou_cost = 1.0 - self.compute_iou(prev_inst['mask'], curr_inst['mask'])
                    flow_cost = self.compute_flow_cost(prev_inst, curr_inst, flow_data, class_id)

                    # if class_id == 4:
                    #     print(iou_cost, flow_cost)

                    # Combine costs
                    combined_cost = (self.iou_weight * iou_cost) + (self.flow_weight * flow_cost)

                    # Use only IoU cost if flow is not reliable
                    if flow_data is None:
                        combined_cost = 1.0 * iou_cost

                    cost_matrix[i, j] = combined_cost

            # Use Hungarian algorithm for this class
            if cost_matrix.size > 0:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)

                # Filter out assignments with high cost
                for row_idx, col_idx in zip(row_indices, col_indices):
                    cost = cost_matrix[row_idx, col_idx]

                    # Use original indices for the assignment
                    orig_prev_idx = prev_indices[row_idx]
                    orig_curr_idx = curr_indices[col_idx]

                    # Only keep if cost is below threshold
                    if cost < 1.0 - self.iou_threshold:
                        all_assignments.append((orig_prev_idx, orig_curr_idx))
                        if orig_curr_idx in all_unassigned_curr:
                            all_unassigned_curr.remove(orig_curr_idx)

        return all_assignments, all_unassigned_curr

    def update_tracks(self,
                      assignments, unassigned_curr, prev_instances,
                      curr_instances, frame_idx, depth_map, flow_data
        ):
        """Update track states based on assignments"""
        # Update matched tracks
        for prev_idx, curr_idx in assignments:
            prev_inst = prev_instances[prev_idx]
            curr_inst = curr_instances[curr_idx]

            track_id = prev_inst.get('track_id')
            if track_id is None:
                continue

            # Update track with new information
            if track_id in self.tracks:
                # Compute 3D coordinates
                depth_info = self.compute_depth_info(curr_inst, depth_map)

                # Update the track
                self.tracks[track_id]['frames'][frame_idx] = {
                    'instance_id': curr_inst['instance_id'],
                    'bbox': curr_inst['bbox'],
                    'centroid': curr_inst['centroid'],
                    'mask': curr_inst['mask'],
                    '3d_cord': depth_info[:3] if depth_info is not None else None,
                    'depth_centroid': depth_info[3:] if depth_info is not None else None,
                    'lost_count': 0,
                    'type': curr_inst.get('type'),
                    'color': curr_inst.get('color'),
                    'pred_type': curr_inst.get('pred_type', {}),
                    'pred_color': curr_inst.get('pred_color', {}),
                    'area': curr_inst.get('area', 0)
                }

                if self.tracks[track_id]['frames'][frame_idx-1]['lost_count'] > 0:
                    for lf_i in range(self.tracks[track_id]['frames'][frame_idx-1]['lost_count']):
                        self.tracks[track_id]['frames'][frame_idx - 1 - lf_i]['lost_count'] = 0

                # Also update the instance with track_id for next frame
                curr_instances[curr_idx]['track_id'] = track_id

        # Create new tracks for unassigned current instances
        for curr_idx in unassigned_curr:
            curr_inst = curr_instances[curr_idx]

            # Compute 3D coordinates
            depth_info = self.compute_depth_info(curr_inst, depth_map)

            # Create new track
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = {
                'class_id': curr_inst['class_id'],
                'frames': {
                    frame_idx: {
                        'instance_id': curr_inst['instance_id'],
                        'bbox': curr_inst['bbox'],
                        'centroid': curr_inst['centroid'],
                        'mask': curr_inst['mask'],
                        '3d_cord': depth_info[:3] if depth_info is not None else None,
                        'depth_centroid': depth_info[3:] if depth_info is not None else None,
                        'lost_count': 0,
                        'type': curr_inst.get('type'),
                        'color': curr_inst.get('color'),
                        'pred_type': curr_inst.get('pred_type', {}),
                        'pred_color': curr_inst.get('pred_color', {}),
                        'area': curr_inst.get('area', 0)
                    }
                },
                'type': curr_inst.get('type'),
                'color': curr_inst.get('color'),
                'pred_type': curr_inst.get('pred_type', {}),
                'pred_color': curr_inst.get('pred_color', {})
            }

            self.tracks[track_id].keys()

            # Update the instance with track_id for next frame
            curr_instances[curr_idx]['track_id'] = track_id

        # Handle lost tracks (not matched in current frame)
        all_matched_tracks = {prev_instances[prev_idx].get('track_id') for prev_idx, _ in assignments}

        dummy_instances = []
        for track_id, track_info in self.tracks.items():
            if track_id not in all_matched_tracks:
                # Track was not matched in current frame
                latest_frame = max(track_info['frames'].keys())

                if latest_frame < frame_idx:
                    # This track was not updated in current frame
                    if latest_frame == frame_idx - 1:
                        # It was active in the previous frame
                        prev_frame_data = track_info['frames'][latest_frame]
                        lost_count = prev_frame_data.get('lost_count', 0) + 1

                        if lost_count <= self.max_lost_frames:
                            # Use flow to predict the new mask and position
                            prev_mask = prev_frame_data['mask']
                            prev_centroid = prev_frame_data['centroid']

                            if flow_data is not None:
                                # Create new mask based on flow predictions
                                new_mask = np.zeros_like(prev_mask, dtype=np.uint8)
                                new_bbox = list(prev_frame_data['bbox'])
                                new_centroid = list(prev_centroid)

                                pred_mask, new_points = self.predict_mask_from_flow(
                                    prev_mask, flow_data, track_info['class_id']
                                )

                                # new_x, new_y = new_points[:, 0], new_points[:, 1]

                                # For each point in the previous mask, find where it moved to
                                # y_indices, x_indices = np.where(prev_mask)
                                # if len(y_indices) > 0:
                                #     # Sample points to avoid processing every pixel
                                #     sample_indices = np.linspace(0, len(y_indices) - 1, min(100, len(y_indices)),
                                #                                  dtype=int)
                                #     sampled_points = [(x_indices[i], y_indices[i]) for i in sample_indices]
                                #
                                #     # Find flow for each sampled point
                                #     new_points = []
                                #     for x, y in sampled_points:
                                #         # Find closest point in flow data
                                #         flow_y, flow_x = y // 10, x // 10  # Adjust based on your flow data sampling
                                #
                                #         if 0 <= flow_y < flow_data.shape[0] and 0 <= flow_x < flow_data.shape[1]:
                                #             # Get the corresponding point in the current frame
                                #             new_pos = flow_data[flow_y, flow_x, 4:6]
                                #             new_x, new_y = int(round(new_pos[0])), int(round(new_pos[1]))
                                #
                                #             if 0 <= new_y < new_mask.shape[0] and 0 <= new_x < new_mask.shape[1]:
                                #                 new_points.append((new_x, new_y))

                                    # Update the bbox based on the new points
                                if new_points.shape[0] > 0:
                                    new_x_coords, new_y_coords = zip(*new_points)
                                    min_x, max_x = min(new_x_coords), max(new_x_coords)
                                    min_y, max_y = min(new_y_coords), max(new_y_coords)

                                    # Update bbox
                                    new_bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

                                    # Create a convex hull mask from the new points
                                    hull_points = np.array(new_points)
                                    if len(hull_points) >= 3:  # Need at least 3 points for convex hull
                                        hull = cv2.convexHull(hull_points)
                                        cv2.fillConvexPoly(new_mask, hull, 1)
                                    else:
                                        # Not enough points for convex hull, use a rectangular mask
                                        cv2.rectangle(new_mask, (min_x, min_y), (max_x, max_y), 1, -1)

                                    # Update centroid
                                    new_centroid = [np.mean(new_x_coords), np.mean(new_y_coords)]

                                    new_mask = new_mask.astype(bool)

                                # Update 3D coordinates if depth information is available
                                new_3d_cord = None
                                new_depth_centroid = None

                                if depth_map is not None:
                                    cx, cy = int(round(new_centroid[0])), int(round(new_centroid[1]))
                                    if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1]:
                                        depth = depth_map[cy, cx]
                                        if depth > 0:
                                            x_3d = (cx - CX_DEPTH) * depth / FX_DEPTH
                                            y_3d = (cy - CY_DEPTH) * depth / FY_DEPTH
                                            z_3d = depth
                                            new_3d_cord = [x_3d, y_3d, z_3d]
                                            new_depth_centroid = [cx, cy]

                                # Save the updated track information
                                self.tracks[track_id]['frames'][frame_idx] = {
                                    'instance_id': prev_frame_data['instance_id'],
                                    'bbox': tuple(new_bbox),
                                    'centroid': tuple(new_centroid),
                                    'mask': new_mask,
                                    '3d_cord': new_3d_cord if new_3d_cord else prev_frame_data.get('3d_cord'),
                                    'depth_centroid': new_depth_centroid if new_depth_centroid else prev_frame_data.get(
                                        'depth_centroid'),
                                    'lost_count': lost_count,
                                    'type': prev_frame_data.get('type'),
                                    'color': prev_frame_data.get('color'),
                                    'pred_type': prev_frame_data.get('pred_type', {}),
                                    'pred_color': prev_frame_data.get('pred_color', {}),
                                    'area': np.count_nonzero(new_mask)
                                }

                                # if frame_idx < 30 and track_info['class_id'] == 10:
                                #     print(
                                #         track_id,
                                #         new_points.shape,
                                #         new_mask.shape,
                                #         np.sum(new_mask.astype(np.uint8)),
                                #         prev_frame_data['instance_id'],
                                #         tuple(new_bbox),
                                #         new_depth_centroid,
                                #         np.count_nonzero(new_mask)
                                #     )

                                dummy_instances.append({
                                    'track_id': track_id,
                                    'class_id': track_info['class_id'],
                                    'instance_id': 'N/A',  # prev_frame_data['instance_id'],
                                    'mask': new_mask,
                                    'bbox': tuple(new_bbox),
                                    'centroid': new_depth_centroid if new_depth_centroid else prev_frame_data.get(
                                        'depth_centroid'),
                                    'area': np.sum(new_mask.astype(np.uint8)),
                                    'frame_idx': frame_idx,
                                    'type': 'N/A',
                                    'color': 'N/A',
                                    'pred_type': 'N/A',
                                    'pred_color': 'N/A'
                                })
                            else:
                                # Fall back to previous frame data if flow is not available
                                self.tracks[track_id]['frames'][frame_idx] = {
                                    'instance_id': prev_frame_data['instance_id'],
                                    'bbox': prev_frame_data['bbox'],
                                    'centroid': prev_frame_data['centroid'],
                                    'mask': prev_frame_data['mask'],
                                    '3d_cord': prev_frame_data.get('3d_cord'),
                                    'depth_centroid': prev_frame_data.get('depth_centroid'),
                                    'lost_count': lost_count,
                                    'type': prev_frame_data.get('type'),
                                    'color': prev_frame_data.get('color'),
                                    'pred_type': prev_frame_data.get('pred_type', {}),
                                    'pred_color': prev_frame_data.get('pred_color', {}),
                                    'area': prev_frame_data.get('area', 0)
                                }
                                dummy_instances.append({
                                    'track_id': track_id,
                                    'class_id': track_info['class_id'],
                                    'instance_id': 'N/A',  # prev_frame_data['instance_id'],
                                    'mask': prev_frame_data['mask'],
                                    'bbox': prev_frame_data['bbox'],
                                    'centroid': prev_frame_data.get('depth_centroid'),
                                    'area': np.sum(prev_frame_data['mask'].astype(np.uint8)),
                                    'frame_idx': frame_idx,
                                    'type': 'N/A',
                                    'color': 'N/A',
                                    'pred_type': 'N/A',
                                    'pred_color': 'N/A'
                                })

        return dummy_instances
        # for track_id, track_info in self.tracks.items():
        #     if track_id not in all_matched_tracks:
        #         # Track was not matched in current frame
        #         latest_frame = max(track_info['frames'].keys())
        #
        #         if latest_frame < frame_idx:
        #             # This track was not updated in current frame
        #             if latest_frame == frame_idx - 1:
        #                 # It was active in the previous frame
        #                 prev_frame_data = track_info['frames'][latest_frame]
        #                 lost_count = prev_frame_data.get('lost_count', 0) + 1
        #
        #                 if lost_count <= self.max_lost_frames:
        #                     # Not lost for too long, propagate it
        #                     # We could use flow here to predict, but simplifying to just keep last position
        #                     # print(prev_frame_data)
        #                     track_info['frames'][frame_idx] = {
        #                         'bbox': prev_frame_data['bbox'],
        #                         'centroid': prev_frame_data['centroid'],
        #                         'mask': prev_frame_data['mask'],
        #                         '3d_cord': prev_frame_data.get('3d_cord'),
        #                         'depth_centroid': prev_frame_data.get('depth_centroid'),
        #                         'lost_count': lost_count,
        #                         'type': prev_frame_data.get('type'),
        #                         'color': prev_frame_data.get('color'),
        #                         'pred_type': prev_frame_data.get('pred_type', {}),
        #                         'pred_color': prev_frame_data.get('pred_color', {}),
        #                         'area': prev_frame_data.get('area', 0)
        #                     }

    # def get_depth_at_point(self, y, x, frame_idx, depth_map_paths):
    #     """Get depth value at a specific pixel location"""
    #     # This function would need to be implemented to access the depth maps
    #     # For example, you might have stored depth maps or have a way to load them
    #     # Return depth value or None if not available
    #
    #     # Example implementation if depth maps are available:
    #     depth_map = self.load_depth(depth_map_paths[frame_idx])
    #     return depth_map[y, x] if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1] else None
    #     # return None  # Placeholder
    #
    def pixel_to_3d(self, x, y, depth):
        """Convert pixel coordinates and depth to 3D world coordinates"""
        # Using the camera intrinsics
        x_3d = (x - CX_DEPTH) * depth / FX_DEPTH
        y_3d = (y - CY_DEPTH) * depth / FY_DEPTH
        z_3d = depth
        return [x_3d, y_3d, z_3d]
    #
    # def find_corresponding_point(self, point_x, point_y, flow_data):
    #     """Find where a point from frame 1 moved to in frame 2 using flow data"""
    #     if flow_data is None or len(flow_data) == 0:
    #         return None
    #
    #     # Find the closest point in the flow data
    #     distances = np.sqrt((flow_data[:, 0] - point_x) ** 2 + (flow_data[:, 1] - point_y) ** 2)
    #     closest_idx = np.argmin(distances)
    #
    #     # Get the corresponding location in frame 2
    #     # flow_data format: [x1, y1, x1_pred, y1_pred, x2, y2, visibility]
    #     corresponding_point = [int(flow_data[closest_idx, 4]), int(flow_data[closest_idx, 5])]
    #
    #     return corresponding_point
    #
    # def get_flow_data(self, frame_idx, next_frame_idx, flow_map_paths):
    #     flow_file = flow_map_paths[frame_idx]
    #     flow_data = self.load_flow(flow_file)
    #     return flow_data
    #

    def compute_object_displacement(self, frame_idx, next_frame_idx, vid_name, keypoints_dir=None):
        """
        Compute displacement of moving objects between two frames using multilateration
        with pre-computed static anchor points, using full 3D coordinates.

        Args:
            frame_idx: Index of first frame
            next_frame_idx: Index of second frame
            depth_map_paths: Optional paths to depth maps (not needed if using keypoints)
            keypoints_dir: Directory containing pre-computed keypoint files

        Returns:
            Dictionary of object displacements by track_id
        """
        try:
            import localization as lx
            import math
        except ImportError:
            print("Error: localization library not found. Please install it with: pip install localization")
            return {}

        displacements = {}

        # Get all tracks active in both frames
        active_tracks = {}
        for track_id, track_info in self.tracks.items():
            if frame_idx in track_info['frames'] and next_frame_idx in track_info['frames']:
                if track_info['frames'][frame_idx].get('lost_count', 0) == 0 and \
                        track_info['frames'][next_frame_idx].get('lost_count', 0) == 0:
                    active_tracks[track_id] = track_info

        if not active_tracks:
            print(f"No active tracks found in both frames {frame_idx} and {next_frame_idx}")
            return {}

        # Load pre-computed keypoints
        if DATASET == 'KITTI':
            keypoints_file = os.path.join(keypoints_dir, f"keypoints-{vid_name}_{next_frame_idx:06d}.npy")
        else:
            keypoints_file = os.path.join(keypoints_dir, f"keypoints-{vid_name}_{next_frame_idx:05d}.npy")

        if not os.path.exists(keypoints_file):
            print(keypoints_file)
            print(f"No keypoints file found for frame {next_frame_idx}")
            return {}

        # Load keypoints - format is [x1, y1, depth1, x2, y2, depth2]
        keypoints = np.load(keypoints_file)

        if len(keypoints) < 4:  # Need at least 4 anchor points for 3D multilateration
            print(f"Not enough keypoints found for frame {next_frame_idx}. Minimum required: 4")
            return {}

        # Process each moving object (vehicles and pedestrians)
        for track_id, track_info in active_tracks.items():
            # Skip static objects as we're only interested in moving objects
            if track_info['class_id'] not in [4, 10]:  # Not a pedestrian or vehicle
                continue

            # Get object data from both frames
            frame1_data = track_info['frames'][frame_idx]
            frame2_data = track_info['frames'][next_frame_idx]

            # Skip if we don't have 3D coordinates
            if frame1_data.get('3d_cord') is None or frame2_data.get('3d_cord') is None:
                continue

            # Get target object coordinates
            target_coord1 = frame1_data['3d_cord']
            target_coord2 = frame2_data['3d_cord']

            # Setup multilateration project using 3D mode
            world = lx.Project(mode='2D', solver='LSE')

            # Add targets for both frames
            target_prev, _ = world.add_target()
            target_curr, _ = world.add_target()

            # Count valid anchors used
            valid_anchors = 0

            # Use keypoints as anchor points
            # Limit to a reasonable number of anchors (e.g., 10)
            max_anchors = min(10, len(keypoints))

            # If we have many keypoints, pick a well-distributed subset
            if len(keypoints) > max_anchors:
                # Sample indices to get a distributed set
                sample_indices = np.linspace(0, len(keypoints) - 1, max_anchors, dtype=int)
                selected_keypoints = keypoints[sample_indices]
            else:
                selected_keypoints = keypoints

            # Add each keypoint as an anchor
            for i, keypoint in enumerate(selected_keypoints):
                # Extract keypoint data
                x1, y1, depth1, x2, y2, depth2 = keypoint

                # Convert to 3D coordinates
                anchor_coord_3d_1 = self.pixel_to_3d(x1, y1, depth1)
                anchor_coord_3d_2 = self.pixel_to_3d(x2, y2, depth2)

                # Add anchor to world (using full 3D coordinates)
                anchor_key = f'anchor_{i}'
                # print([anchor_coord_3d_1[0], anchor_coord_3d_1[2]])
                world.add_anchor(anchor_key, [anchor_coord_3d_1[0], anchor_coord_3d_1[2]])

                # Calculate 3D distances from this anchor to target in both frames
                prev_dist = math.dist(
                    [target_coord1[0], target_coord1[2]],
                    [anchor_coord_3d_1[0], anchor_coord_3d_1[2]]
                )
                curr_dist = math.dist(
                    [target_coord2[0], target_coord2[2]],
                    [anchor_coord_3d_2[0], anchor_coord_3d_2[2]]
                )

                # print(prev_dist, curr_dist)

                # Add measurements
                target_prev.add_measure(anchor_key, prev_dist)
                target_curr.add_measure(anchor_key, curr_dist)

                valid_anchors += 1

            # Check if we have enough anchors
            if valid_anchors < 3:  # 2D multilateration needs at least 4 non-coplanar points
                print(f"Not enough valid anchors for track {track_id}. Skipping.")
                continue

            try:
                # Solve the multilateration problem
                world.solve()

                # print(target_prev.loc)

                # Extract locations
                loc_prev_str = str(target_prev.loc).replace('p(', '').replace(')', '')
                loc_curr_str = str(target_curr.loc).replace('p(', '').replace(')', '')

                # Parse coordinates (handle different string formats from the library)
                try:
                    loc_prev = [float(coord) for coord in loc_prev_str.split(',')[:-1]]
                    loc_curr = [float(coord) for coord in loc_curr_str.split(',')[:-1]]
                except:
                    # Alternative parsing if the above fails
                    loc_prev = [float(coord) for coord in loc_prev_str.split(' ') if coord]
                    loc_curr = [float(coord) for coord in loc_curr_str.split(' ') if coord]

                # print(loc_prev, loc_curr)

                # Calculate full 3D displacement
                displacement = math.dist(loc_prev, loc_curr)

                # Store the result
                displacements[track_id] = {
                    'from_frame': frame_idx,
                    'to_frame': next_frame_idx,
                    'displacement': displacement,
                    'loc_prev': loc_prev,
                    'loc_curr': loc_curr,
                    'class_id': track_info['class_id'],
                    'class_name': self.get_class_name(track_info['class_id'])
                }

            except Exception as e:
                print(f"Error solving multilateration for track {track_id}: {e}")

        return displacements

    def save_tracks_by_class(self, output_dir, video_name, video_dir):
        """Save tracks organized by class as pickle files"""
        os.makedirs(output_dir, exist_ok=True)

        print("Computing displacements for all tracks...")

        # Get all frames in order
        all_frames = set()
        for track_id, track_info in self.tracks.items():
            all_frames.update(track_info['frames'].keys())
        all_frames = sorted(list(all_frames))

        # Calculate displacements between consecutive frames
        keypoints_dir = os.path.join(video_dir, "keypoints")
        for i in range(len(all_frames) - 1):
            frame_idx = all_frames[i]
            next_frame_idx = all_frames[i + 1]

            # Compute displacements
            displacements = self.compute_object_displacement(frame_idx, next_frame_idx, video_name, keypoints_dir=keypoints_dir)

            # Update displacement information in the track data
            for track_id, displacement_info in displacements.items():
                if track_id in self.tracks:
                    # Update the current frame with the displacement
                    curr_frame = displacement_info['to_frame']
                    if curr_frame in self.tracks[track_id]['frames']:
                        self.tracks[track_id]['frames'][curr_frame]['displacement'] = displacement_info['displacement']

        # Group tracks by class
        tracks_by_class = defaultdict(dict)

        for track_id, track_info in self.tracks.items():
            class_id = track_info['class_id']
            class_name = self.get_class_name(class_id)

            # Skip if class name is not one we're interested in
            if class_name is None:
                continue

            # Convert track format to match required output format
            track_by_frame = {}

            for frame_idx, frame_data in track_info['frames'].items():
                # Skip if it's a lost frame
                if frame_data.get('lost_count', 0) > 0:
                    continue

                # Get frame ID in the required format
                if DATASET == 'KITTI':
                    frame_num = f"{int(frame_idx):06d}"
                else:
                    frame_num = f"{int(frame_idx):05d}"

                # Store data in the required format
                track_by_frame[frame_num] = {
                    "b_box": [
                        frame_data['bbox'][1],
                        frame_data['bbox'][0],
                        frame_data['bbox'][1] + frame_data['bbox'][3],
                        frame_data['bbox'][0] + frame_data['bbox'][2]
                    ],
                    "3d_cord": [float(crd) for crd in frame_data.get('3d_cord')] if frame_data.get('3d_cord') else None,
                    "centroid": frame_data.get('depth_centroid'),
                    "displacement": frame_data.get('displacement')
                }

            # Add to tracks by class if it has at least one valid frame
            if track_by_frame:
                tracks_by_class[class_name][track_id] = track_by_frame

        # Save each class to a separate pickle file
        for class_name, tracks in tracks_by_class.items():
            if not tracks:
                continue

            output_path = os.path.join(output_dir, f"trackers_{video_name}_{class_name}.pickle")
            # print(tracks)
            with open(output_path, 'wb') as f:
                pickle.dump(tracks, f)

            print(f"Saved {len(tracks)} tracks for class {class_name} to {output_path}")

        vehicle_info_path = os.path.join(output_dir, f"vehicle_info_{video_name}.txt")
        model_info_path = os.path.join(output_dir, f"model_vehicle_info_{video_name}.txt")
        model_info_path_best = os.path.join(output_dir, f"model_vehicle_info_best_{video_name}.txt")
        model_frequent_path = os.path.join(output_dir, f"model_vehicle_info_frequent_{video_name}.txt")
        normal_frequent_path = os.path.join(output_dir, f"normal_vehicle_info_frequent_{video_name}.txt")

        # New combined results file
        combined_info_path = os.path.join(output_dir, f"combined_vehicle_info_{video_name}.txt")

        with open(vehicle_info_path, 'w') as f_normal, open(model_info_path, 'w') as f_model, \
                open(model_info_path_best, 'w') as f_model_best, open(model_frequent_path, 'w') as f_model_freq, \
                open(normal_frequent_path, 'w') as f_normal_freq, open(combined_info_path, 'w') as f_combined:

            for track_id, track_info in self.tracks.items():
                class_id = track_info['class_id']
                if class_id == 10:  # vehicles
                    # Find the frame with maximum area for normal predictions
                    max_area_frame = None
                    max_area = 0
                    best_type = "NA"
                    best_color = "NA"

                    # Find the frame with highest model confidence
                    best_type_conf = 0
                    best_color_conf = 0
                    model_best_type = "NA"
                    model_best_color = "NA"

                    model_type = "NA"
                    model_color = "NA"

                    # Counters for most frequent detections
                    normal_type_counter = {}
                    normal_color_counter = {}
                    model_type_counter = {}
                    model_color_counter = {}

                    # Loop through all frames for this track
                    for frame_idx, frame_data in track_info['frames'].items():
                        # Skip if it's a lost frame
                        if frame_data.get('lost_count', 0) > 0:
                            continue

                        # Count normal detections for frequency
                        normal_type = frame_data.get('type', "NA")
                        if normal_type != "NA":
                            normal_type_counter[normal_type] = normal_type_counter.get(normal_type, 0) + 1

                        normal_color = frame_data.get('color', "NA")
                        if normal_color != "NA":
                            normal_color_counter[normal_color] = normal_color_counter.get(normal_color, 0) + 1

                        # Check for maximum area (normal prediction)
                        area = frame_data.get('area', 0)
                        if area > max_area:
                            max_area = area
                            max_area_frame = frame_idx
                            best_type = normal_type
                            best_color = normal_color

                            pred_type = frame_data.get('pred_type', {})
                            if pred_type:
                                type_item = max(pred_type.items(), key=lambda x: x[1])
                                model_type = type_item[0]

                            pred_color = frame_data.get('pred_color', {})
                            if pred_color:
                                color_item = max(pred_color.items(), key=lambda x: x[1])
                                model_color = color_item[0]

                        # Check for best model prediction confidence
                        pred_type = frame_data.get('pred_type', {})
                        if pred_type:
                            # Count model detections for frequency
                            top_type = max(pred_type.items(), key=lambda x: x[1])[0]
                            model_type_counter[top_type] = model_type_counter.get(top_type, 0) + 1

                            # Get the highest confidence and its type
                            type_item = max(pred_type.items(), key=lambda x: x[1])
                            if type_item[1] > best_type_conf:
                                best_type_conf = type_item[1]
                                model_best_type = type_item[0]

                        pred_color = frame_data.get('pred_color', {})
                        if pred_color:
                            # Count model detections for frequency
                            top_color = max(pred_color.items(), key=lambda x: x[1])[0]
                            model_color_counter[top_color] = model_color_counter.get(top_color, 0) + 1

                            # Get the highest confidence and its color
                            color_item = max(pred_color.items(), key=lambda x: x[1])
                            if color_item[1] > best_color_conf:
                                best_color_conf = color_item[1]
                                model_best_color = color_item[0]

                    # Find most frequent normal detections
                    most_frequent_normal_type = max(normal_type_counter.items(), key=lambda x: x[1])[
                        0] if normal_type_counter else "NA"
                    most_frequent_normal_color = max(normal_color_counter.items(), key=lambda x: x[1])[
                        0] if normal_color_counter else "NA"

                    # Find most frequent model detections
                    most_frequent_model_type = max(model_type_counter.items(), key=lambda x: x[1])[
                        0] if model_type_counter else "NA"
                    most_frequent_model_color = max(model_color_counter.items(), key=lambda x: x[1])[
                        0] if model_color_counter else "NA"

                    all_model_types = [model_type, model_best_type, most_frequent_model_type]
                    model_type_counter_combined = {}
                    for t in all_model_types:
                        if t != "NA":
                            model_type_counter_combined[t] = model_type_counter_combined.get(t, 0) + 1

                    # Get most common model type, fallback to first non-NA value if all are different
                    if model_type_counter_combined:
                        most_common_model_type = max(model_type_counter_combined.items(), key=lambda x: x[1])[0]
                    else:
                        most_common_model_type = next((t for t in all_model_types if t != "NA"), "NA")

                    # Get normal type (already have best_type from normal detector)
                    normal_type_combined = best_type  # Using type from max area frame

                    # For color - consider all 5 approaches
                    all_colors = [best_color, model_color, model_best_color,
                                  most_frequent_model_color, most_frequent_normal_color]
                    color_counter_combined = {}
                    for c in all_colors:
                        if c != "NA":
                            color_counter_combined[c] = color_counter_combined.get(c, 0) + 1

                    # Get most common color across all approaches
                    if color_counter_combined:
                        most_common_color = max(color_counter_combined.items(), key=lambda x: x[1])[0]
                    else:
                        most_common_color = next((c for c in all_colors if c != "NA"), "NA")

                    # Write to normal file (based on area)
                    f_normal.write(f"TypeOf(Vehicles_{track_id}, {best_type})\n")
                    f_normal.write(f"ColorOf(Vehicles_{track_id}, {best_color})\n")

                    # Write to model file (based on frame with max area)
                    f_model.write(f"TypeOf(Vehicles_{track_id}, {model_type})\n")
                    f_model.write(f"ColorOf(Vehicles_{track_id}, {model_color})\n")

                    # Write to model best file (based on highest confidence)
                    f_model_best.write(f"TypeOf(Vehicles_{track_id}, {model_best_type})\n")
                    f_model_best.write(f"ColorOf(Vehicles_{track_id}, {model_best_color})\n")

                    # Write to model frequent file (most frequent model detections)
                    f_model_freq.write(f"TypeOf(Vehicles_{track_id}, {most_frequent_model_type})\n")
                    f_model_freq.write(f"ColorOf(Vehicles_{track_id}, {most_frequent_model_color})\n")

                    # Write to normal frequent file (most frequent normal detections)
                    f_normal_freq.write(f"TypeOf(Vehicles_{track_id}, {most_frequent_normal_type})\n")
                    f_normal_freq.write(f"ColorOf(Vehicles_{track_id}, {most_frequent_normal_color})\n")

                    # Write to combined file
                    f_combined.write(f"TypeOf(Vehicles_{track_id}, {most_common_model_type})\n")
                    f_combined.write(f"TypeOf(Vehicles_{track_id}, {normal_type_combined})\n")
                    f_combined.write(f"ColorOf(Vehicles_{track_id}, {most_common_color})\n")

        print(f"Saved vehicle type and color information to {vehicle_info_path}")
        print(f"Saved model-based vehicle type and color information to {model_info_path}")
        print(f"Saved model-based best vehicle type and color information to {model_info_path_best}")
        print(f"Saved most frequent model detection information to {model_frequent_path}")
        print(f"Saved most frequent normal detection information to {normal_frequent_path}")
        print(f"Saved combined vehicle information to {combined_info_path}")

    def get_class_name(self, class_id):
        """Get class name from class ID"""
        classes = {
            0: None,  # unlabeled
            1: "building",
            2: "fence",
            3: None,  # other
            4: "pedestrian",
            5: "pole",
            6: "roadline",
            7: "road",
            8: "sidewalk",
            9: "vegetation",
            10: "vehicles",
            11: "wall",
            12: "trafficsign",
            13: "sky",
            14: None,  # ground
            15: "bridge",
            16: "railtrack",
            17: "guardrail",
            18: "trafficlight",
            19: None,  # static
            20: None,  # dynamic
            21: "water",
            22: "terrain"
        }
        return classes.get(class_id)


def process_video(video_dir, output_dir, video_name, rgb_base_dir=None, type_model_path=None, color_model_path=None):
    """Process a single video sequence"""
    print(f"Processing video: {video_name}")

    rgb_dir = os.path.join(rgb_base_dir, video_name) if rgb_base_dir else None

    # Set paths
    panoptic_dir = os.path.join(video_dir, "label")
    depth_dir = os.path.join(video_dir, "depth")
    flow_dir = os.path.join(video_dir, "flow")

    # Create output directory
    os.makedirs(os.path.join(output_dir, video_name), exist_ok=True)

    # Find all panoptic label files
    label_files = natsorted(glob.glob(os.path.join(panoptic_dir, "*.png")))
    if not label_files:
        print(f"No label files found in {panoptic_dir}")
        return

    # Initialize tracker
    tracker = InstanceTracker(
        max_lost_frames=5,
        flow_weight=0.75,
        iou_threshold=0.1,
        min_obj_size=10,
        type_model_path=type_model_path,
        color_model_path=color_model_path
    )

    # Keep track of instances from previous frame
    prev_instances = []
    depth_map_paths = []
    flow_map_paths = []

    # Process frames
    for frame_idx, label_file in enumerate(tqdm(label_files)):
        # Get frame number from filename
        frame_name = os.path.basename(label_file).split('.')[0]
        try:
            frame_num = int(frame_name.split('_')[-1])
        except ValueError:
            # Handle case where frame names are not just numbers
            frame_num = frame_idx

        # Load panoptic label
        class_mask, instance_id = tracker.load_label_mask(label_file)

        # Load depth map
        depth_file = os.path.join(depth_dir, f"{frame_name}.png")
        depth_map_paths.append(depth_file)
        depth_map = tracker.load_depth(depth_file, class_mask.shape)

        assert depth_map.shape == class_mask.shape

        # Load flow data if not the first frame
        flow_data = None
        if frame_idx > 0:
            prev_frame_name = os.path.basename(label_files[frame_idx - 1]).split('.')[0]
            if DATASET == 'KITTI':
                flow_file = os.path.join(flow_dir, f"flow-{video_name}_{frame_num:06d}.npy")
            else:
                flow_file = os.path.join(flow_dir, f"flow-{video_name}_{frame_num:05d}.npy")
            flow_data = tracker.load_flow(flow_file)
            flow_map_paths.append(flow_file)

            # print(flow_data)

        rgb_image = None
        if rgb_dir:
            if DATASET == 'KITTI':
                rgb_file = os.path.join(rgb_dir, f"{frame_num:06d}.png")
            else:
                rgb_file = os.path.join(rgb_dir, f"{frame_num:05d}.png")
            if os.path.exists(rgb_file):
                rgb_image = np.array(Image.open(rgb_file))
                # rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Extract instances from the current frame
        # curr_instances = tracker.extract_instances(class_mask, instance_id, frame_num)
        curr_instances = tracker.extract_instances(class_mask, instance_id, frame_num, rgb_image)

        # Track instances
        if prev_instances:
            # Assign current instances to existing tracks
            assignments, unassigned_curr = tracker.assign_tracks(prev_instances, curr_instances, flow_data)

            # Update track states
            dummy_instances = tracker.update_tracks(
                assignments, unassigned_curr, prev_instances, curr_instances, frame_num, depth_map, flow_data
            )

            curr_instances = curr_instances + dummy_instances
        else:
            # First frame, initialize tracks for all instances
            for curr_inst in curr_instances:
                class_id = curr_inst['class_id']
                # Compute 3D coordinates
                depth_info = tracker.compute_depth_info(curr_inst, depth_map)

                # Create new track
                track_id = tracker.next_track_id
                tracker.next_track_id += 1

                tracker.tracks[track_id] = {
                    'class_id': curr_inst['class_id'],
                    'frames': {
                        frame_num: {
                            'instance_id': curr_inst['instance_id'],
                            'bbox': curr_inst['bbox'],
                            'centroid': curr_inst['centroid'],
                            'mask': curr_inst['mask'],
                            '3d_cord': depth_info[:3] if depth_info is not None else None,
                            'depth_centroid': depth_info[3:] if depth_info is not None else None,
                            'lost_count': 0,
                            'type': curr_inst.get('type'),
                            'color': curr_inst.get('color'),
                            'pred_type': curr_inst.get('pred_type', {}),
                            'pred_color': curr_inst.get('pred_color', {}),
                            'area': curr_inst.get('area', 0)
                        }
                    },
                    'type': curr_inst.get('type'),
                    'color': curr_inst.get('color'),
                    'pred_type': curr_inst.get('pred_type', {}),
                    'pred_color': curr_inst.get('pred_color', {})
                }

                # Add track_id to the instance
                curr_inst['track_id'] = track_id

        # Current becomes previous for next iteration
        prev_instances = curr_instances

    # Save tracks
    tracker.save_tracks_by_class(os.path.join(output_dir, video_name), video_name, video_dir)

    print(f"Completed processing video: {video_name}")


def process_video_parallel(args):
    """Wrapper function for process_video to use with multiprocessing"""
    video_dir, output_dir, video_name, rgb_base_dir, type_model_path, color_model_path = args
    process_video(
        video_dir, output_dir, video_name, rgb_base_dir,
        type_model_path=type_model_path,
        color_model_path=color_model_path
    )
    return f"Completed processing {video_name}"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Track instances in panoptic segmentation using flow and depth")
    parser.add_argument("--parent_dir", required=True, help="Parent directory containing video folders")
    parser.add_argument("--output_dir", required=True, help="Directory to save tracker outputs")
    parser.add_argument("--rgb_base_dir", default="/data/datasets/KITTI/STEP/testing/image_02",
                        help="Base directory for RGB images")
    parser.add_argument("--type_model_path", default=None, help="Path to vehicle type model")
    parser.add_argument("--color_model_path", default=None, help="Path to vehicle color model")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of processes to use (default: number of CPU cores)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Find all video directories
    video_dirs = natsorted([f.path for f in os.scandir(args.parent_dir) if f.is_dir()])[:15]  # [5:]  # [20:]

    # print(video_dirs)

    if not video_dirs:
        print(f"No video directories found in {args.parent_dir}")
        return

    # Process each video
    # for video_dir in video_dirs:
    #     video_name = os.path.basename(video_dir)
    #     process_video(
    #         video_dir,
    #         args.output_dir,
    #         video_name,
    #         args.rgb_base_dir,
    #         args.type_model_path,
    #         args.color_model_path
    #     )
    #
    # print(f"Completed processing {len(video_dirs)} videos")

    # Determine number of processes
    num_processes = args.num_processes or multiprocessing.cpu_count()
    print(f"Using {num_processes} processes for parallel processing")

    # Prepare arguments for each video
    process_args = []
    for video_dir in video_dirs:
        video_name = os.path.basename(video_dir)
        process_args.append((video_dir, args.output_dir, video_name, args.rgb_base_dir,
                             args.type_model_path, args.color_model_path))

    # Process videos in parallel
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(process_video_parallel, process_args)

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
