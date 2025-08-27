"""
BEAT2 Face Template Generator

This script processes BEAT2 datasets from multiple languages (Chinese, English, Japanese, Spanish)
to generate averaged face templates using the FLAME model. It:

1. Loads test split data from all four BEAT2 language datasets
2. Extracts facial expressions and jaw poses from SMPLX parameters
3. Processes the data through FLAME model to compute vertices in two versions:
   - With global rotation (using actual global orientation from data)
   - Without global rotation (global orientation set to zero)
4. Averages all vertices across all frames to create neutral face templates
5. Saves both templates for later use in face animation tasks

The script generates two template files:
- Template with global rotation: Contains vertices computed with actual global orientation
- Template without global rotation: Contains vertices computed with zero global orientation

Usage:
    # Generate both templates (high memory usage):
    python beat2_face_template_generator.py --flame_path model_files/FLAME2020/ --template_type both
    
    # Generate only with global rotation template (saves memory):
    python beat2_face_template_generator.py --flame_path model_files/FLAME2020/ --template_type with_global --output_path face_template.npz
    
    # Generate only without global rotation template (saves memory):
    python beat2_face_template_generator.py --flame_path model_files/FLAME2020/ --template_type without_global --output_path_no_global face_template_no_global.npz

Dependencies:
    - torch
    - numpy
    - pandas
    - tqdm
    - smplx
    - lom (local modules)

Author: Language of Motion Project
"""

import torch
import os
import numpy as np
from os.path import join as pjoin
import argparse
from tqdm import tqdm
import pandas as pd
from smplx import FLAME
from lom.utils.rotation_conversions import axis_angle_to_6d_np
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HANDS,
    BEAT_SMPLX_LOWER
)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Argument parsing
parser = argparse.ArgumentParser('BEAT2 Face Template Generator')
parser.add_argument('--flame_path', type=str, default="model_files/FLAME2020/", 
                    help="Path to the FLAME model files")
parser.add_argument('--output_path', type=str, default="./model_files/face_template.npz",
                    help="Path to save the averaged face template")
parser.add_argument('--output_path_no_global', type=str, default="./model_files/face_template_no_global.npz",
                    help="Path to save the face template without global rotation")
parser.add_argument('--batch_size', type=int, default=64, 
                    help="Batch size for FLAME processing")
parser.add_argument('--template_type', type=str, choices=['with_global', 'without_global', 'both'], 
                    default='both', help="Which template to generate: 'with_global', 'without_global', or 'both'")
args = parser.parse_args()

# Dataset paths for all four languages
dataset_paths = [
    "/data/datasets/BEAT2/beat_chinese_v2.0.0/smplxflame_30",
    "/data/datasets/BEAT2/beat_english_v2.0.0/smplxflame_30", 
    "/data/datasets/BEAT2/beat_japanese_v2.0.0/smplxflame_30",
    "/data/datasets/BEAT2/beat_spanish_v2.0.0/smplxflame_30"
]

# Training speakers list (as used in the dataset loaders)
training_speakers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

def forward_flame(flame_model, exps, jaw, shape, batch_size, global_orient=None):
    """
    Forward pass through FLAME model with batching support.
    Adapted from lom.py forward_flame method.
    
    Args:
        flame_model: FLAME model instance
        exps: Expression parameters (n_frames, 100)
        jaw: Jaw pose parameters (n_frames, 3)
        shape: Shape parameters (n_frames, 10)
        batch_size: Batch size for processing
        global_orient: Global orientation parameters (n_frames, 3). If None, uses zeros.
    """
    # Ensure FLAME model is on the correct device
    if flame_model.J_regressor.device != exps.device:
        flame_model.to(exps.device)
    
    # Set default global orientation to zeros if not provided
    if global_orient is None:
        global_orient = torch.zeros((exps.shape[0], 3), device=exps.device)
    
    actual_length = exps.shape[0]
    if actual_length > batch_size:
        s, r = actual_length//batch_size, actual_length%batch_size
        output_joints = []
        output_vertices = []
        for div_idx in range(s):
            flame_output = flame_model(
                global_orient=global_orient[div_idx*batch_size:(div_idx+1)*batch_size],
                expression=exps[div_idx*batch_size:(div_idx+1)*batch_size],
                jaw_pose=jaw[div_idx*batch_size:(div_idx+1)*batch_size],
                shape=shape[div_idx*batch_size:(div_idx+1)*batch_size],
            )
            output_joints.append(flame_output.joints)
            output_vertices.append(flame_output.vertices)
        if r != 0:
            exps_pad = torch.cat([exps[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 100), device=exps.device)], dim=0)
            jaw_pad = torch.cat([jaw[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 3), device=jaw.device)], dim=0)
            shape_pad = torch.cat([shape[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 10), device=shape.device)], dim=0)
            global_orient_pad = torch.cat([global_orient[s*batch_size:s*batch_size+r], torch.zeros((batch_size-r, 3), device=global_orient.device)], dim=0)
            flame_output = flame_model(
                global_orient=global_orient_pad,
                expression=exps_pad,
                jaw_pose=jaw_pad,
                shape=shape_pad,
            )
            output_joints.append(flame_output.joints[:r])
            output_vertices.append(flame_output.vertices[:r])
    else:
        output_joints = []
        output_vertices = []
        exps_pad = torch.cat([exps, torch.zeros((batch_size-actual_length, 100), device=exps.device)], dim=0)
        jaw_pad = torch.cat([jaw, torch.zeros((batch_size-actual_length, 3), device=jaw.device)], dim=0)
        shape_pad = torch.cat([shape, torch.zeros((batch_size-actual_length, 10), device=shape.device)], dim=0)
        global_orient_pad = torch.cat([global_orient, torch.zeros((batch_size-actual_length, 3), device=global_orient.device)], dim=0)
        flame_output = flame_model(
            global_orient=global_orient_pad,
            expression=exps_pad,
            jaw_pose=jaw_pad,
            shape=shape_pad,
        )
        output_joints.append(flame_output.joints[:actual_length])
        output_vertices.append(flame_output.vertices[:actual_length])

    output_joints = torch.cat(output_joints, dim=0)
    output_vertices = torch.cat(output_vertices, dim=0)

    return output_joints, output_vertices

def load_test_files_from_dataset(dataset_path):
    """
    Load test split files from a specific dataset path.
    Returns list of file paths for test data.
    """
    # Check if split CSV exists
    split_csv_path = pjoin(os.path.dirname(dataset_path), "train_test_split.csv")
    if not os.path.exists(split_csv_path):
        print(f"Warning: train_test_split.csv not found at {split_csv_path}")
        print("Using all available files as test files")
        # If no split file, use all .npz files as test files
        test_files = []
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith(".npz"):
                    test_files.append(pjoin(dataset_path, file))
        return test_files
    
    # Load split rules from CSV file
    split_rule = pd.read_csv(split_csv_path)
    
    # Filter files for test split and training speakers
    selected_file = split_rule.loc[
        (split_rule['type'] == 'test') &
        (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
    ]
    
    test_files = []
    for index, file_name in selected_file.iterrows():
        f_name = file_name["id"]
        pose_file = pjoin(dataset_path, f_name + ".npz")
        if os.path.exists(pose_file):
            test_files.append(pose_file)
        else:
            print(f"Warning: File not found: {pose_file}")
    
    return test_files

def process_face_data(pose_file):
    """
    Process a single pose file to extract face parameters.
    Based on dataset_face_vq.py lines 443-469.
    """
    try:
        # Load pose data from NPZ file
        pose_data = np.load(pose_file, allow_pickle=True)
        poses = pose_data["poses"]
        n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
        betas = pose_data["betas"]
        # Repeat betas for each frame
        betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)
        expressions = pose_data["expressions"]

        # Apply joint mask to filter relevant joints
        pose_processed = poses * JOINT_MASK_FULL
        pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
        
        # Extract different components from processed pose
        tar_pose = pose_processed[:, :165]  # Body pose (55 joints * 3)
        tar_exps = expressions  # Facial expressions

        # Extract global orientation (first 3 elements of the original poses, before masking)
        # global_orient = poses[:, :3]  # Global orientation (3D rotation)
        global_orient = poses[:, 45:48]
        
        # Extract and convert jaw pose data
        tar_pose_jaw = tar_pose[:, 66:69]  # Jaw pose (3D rotation)
        tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)  # Convert to 6D representation
        
        return {
            'expressions': tar_exps,  # (n_frames, 100)
            'jaw_6d': tar_pose_jaw_6d,  # (n_frames, 6)
            'jaw_axis_angle': tar_pose_jaw,  # (n_frames, 3)
            'global_orient': global_orient,  # (n_frames, 3)
            'betas': betas,  # (n_frames, 300)
        }
    except Exception as e:
        print(f"Error processing file {pose_file}: {e}")
        return None

def validate_paths():
    """Validate that required paths exist before processing."""
    # Check FLAME model path
    if not os.path.exists(args.flame_path):
        print(f"Error: FLAME model path not found: {args.flame_path}")
        return False
    
    # Check at least one dataset path exists
    valid_datasets = []
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            valid_datasets.append(dataset_path)
    
    if len(valid_datasets) == 0:
        print("Error: None of the dataset paths exist:")
        for path in dataset_paths:
            print(f"  - {path}")
        return False
    
    print(f"Found {len(valid_datasets)} valid dataset paths:")
    for path in valid_datasets:
        print(f"  - {path}")
    
    return True

def main():
    # Validate paths before starting
    if not validate_paths():
        return
    
    # Initialize FLAME model
    print("Initializing FLAME model...")
    try:
        flame_model = FLAME(
            args.flame_path, 
            num_expression_coeffs=100, 
            ext='pkl', 
            batch_size=args.batch_size
        ).to(device).eval()
    except Exception as e:
        print(f"Error initializing FLAME model: {e}")
        return
    
    print("Loading test files from all datasets...")
    all_test_files = []
    
    # Collect test files from all four datasets
    for dataset_path in dataset_paths:
        if os.path.exists(dataset_path):
            print(f"Processing dataset: {dataset_path}")
            test_files = load_test_files_from_dataset(dataset_path)
            all_test_files.extend(test_files)
            print(f"Found {len(test_files)} test files")
        else:
            print(f"Warning: Dataset path not found: {dataset_path}")
    
    print(f"Total test files found: {len(all_test_files)}")
    
    if len(all_test_files) == 0:
        print("Error: No test files found. Please check the dataset paths.")
        return
    
    # Process all files and collect face data
    print(f"Processing face data from all test files (template type: {args.template_type})...")
    all_vertices_with_global = []
    all_vertices_no_global = []
    total_frames = 0
    
    for pose_file in tqdm(all_test_files, desc="Processing files"):
        face_data = process_face_data(pose_file)
        if face_data is None:
            continue
            
        expressions = face_data['expressions']  # (n_frames, 100)
        jaw_axis_angle = face_data['jaw_axis_angle']  # (n_frames, 3)
        global_orient = face_data['global_orient']  # (n_frames, 3)
        
        # Convert to torch tensors
        expressions_tensor = torch.from_numpy(expressions).float().to(device)
        jaw_tensor = torch.from_numpy(jaw_axis_angle).float().to(device)
        global_orient_tensor = torch.from_numpy(global_orient).float().to(device)
        shape_tensor = torch.zeros((expressions.shape[0], 10), device=device)
        
        # Process based on template type to save memory
        if args.template_type in ['with_global', 'both']:
            # Compute FLAME vertices with global rotation (using actual global orientation)
            with torch.no_grad():
                joints_global, vertices_global = forward_flame(
                    flame_model, 
                    expressions_tensor, 
                    jaw_tensor, 
                    shape_tensor, 
                    args.batch_size,
                    global_orient=global_orient_tensor
                )
            all_vertices_with_global.append(vertices_global.cpu())
        
        if args.template_type in ['without_global', 'both']:
            # Compute FLAME vertices without global rotation (global orientation set to zero)
            global_orient_zero = torch.zeros_like(global_orient_tensor)
            with torch.no_grad():
                joints_no_global, vertices_no_global = forward_flame(
                    flame_model, 
                    expressions_tensor, 
                    jaw_tensor, 
                    shape_tensor, 
                    args.batch_size,
                    global_orient=global_orient_zero
                )
            all_vertices_no_global.append(vertices_no_global.cpu())
        
        total_frames += expressions.shape[0]
    
    # Validate that we have data based on template type
    if args.template_type in ['with_global', 'both'] and len(all_vertices_with_global) == 0:
        print("Error: No valid face data found for with_global template.")
        return
    if args.template_type in ['without_global', 'both'] and len(all_vertices_no_global) == 0:
        print("Error: No valid face data found for without_global template.")
        return
    
    num_files = max(len(all_vertices_with_global), len(all_vertices_no_global))
    print(f"Successfully processed {num_files} files with {total_frames} total frames")
    
    # Compute averages based on template type
    print(f"Computing averages for {args.template_type} template...")
    
    mean_vertices_with_global_np = None
    mean_vertices_no_global_np = None
    
    if args.template_type in ['with_global', 'both']:
        # Concatenate and compute mean for with_global
        all_vertices_with_global_cat = torch.cat(all_vertices_with_global, dim=0)  # (total_frames, n_vertices, 3)
        mean_vertices_with_global = torch.mean(all_vertices_with_global_cat, dim=0)  # (n_vertices, 3)
        mean_vertices_with_global_np = mean_vertices_with_global.numpy()
        print(f"Average vertices with global rotation shape: {mean_vertices_with_global.shape}")
    
    if args.template_type in ['without_global', 'both']:
        # Concatenate and compute mean for without_global
        all_vertices_no_global_cat = torch.cat(all_vertices_no_global, dim=0)      # (total_frames, n_vertices, 3)
        mean_vertices_no_global = torch.mean(all_vertices_no_global_cat, dim=0)     # (n_vertices, 3)
        mean_vertices_no_global_np = mean_vertices_no_global.numpy()
        print(f"Average vertices without global rotation shape: {mean_vertices_no_global.shape}")
    
    # Save templates based on what was computed
    saved_files = []
    
    if args.template_type in ['with_global', 'both'] and mean_vertices_with_global_np is not None:
        # Save the averaged template with global rotation
        print(f"Saving face template with global rotation to: {args.output_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        np.savez(
            args.output_path,
            mean_vertices=mean_vertices_with_global_np,
            total_frames_processed=total_frames,
            num_files_processed=len(all_vertices_with_global),
            dataset_paths=dataset_paths,
            template_type="with_global_rotation"
        )
        saved_files.append(f"Template with global rotation: {args.output_path}")
    
    if args.template_type in ['without_global', 'both'] and mean_vertices_no_global_np is not None:
        # Save the averaged template without global rotation
        print(f"Saving face template without global rotation to: {args.output_path_no_global}")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path_no_global), exist_ok=True)
        
        np.savez(
            args.output_path_no_global,
            mean_vertices=mean_vertices_no_global_np,
            total_frames_processed=total_frames,
            num_files_processed=len(all_vertices_no_global),
            dataset_paths=dataset_paths,
            template_type="without_global_rotation"
        )
        saved_files.append(f"Template without global rotation: {args.output_path_no_global}")
    
    print("Face template generation complete!")
    print(f"Processed {total_frames} frames from {num_files} files")
    for saved_file in saved_files:
        print(f"Saved: {saved_file}")
    
    # Print some statistics
    print("\nTemplate Statistics:")
    if mean_vertices_with_global_np is not None:
        print("With Global Rotation:")
        print(f"  Mean vertices range: [{mean_vertices_with_global_np.min():.4f}, {mean_vertices_with_global_np.max():.4f}]")
        print(f"  Mean vertices center: [{mean_vertices_with_global_np.mean(axis=0)[0]:.4f}, {mean_vertices_with_global_np.mean(axis=0)[1]:.4f}, {mean_vertices_with_global_np.mean(axis=0)[2]:.4f}]")
    
    if mean_vertices_no_global_np is not None:
        print("Without Global Rotation:")
        print(f"  Mean vertices range: [{mean_vertices_no_global_np.min():.4f}, {mean_vertices_no_global_np.max():.4f}]")
        print(f"  Mean vertices center: [{mean_vertices_no_global_np.mean(axis=0)[0]:.4f}, {mean_vertices_no_global_np.mean(axis=0)[1]:.4f}, {mean_vertices_no_global_np.mean(axis=0)[2]:.4f}]")

if __name__ == "__main__":
    main()