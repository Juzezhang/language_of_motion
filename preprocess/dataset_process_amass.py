"""
AMASS Dataset Processing Script

This script processes AMASS motion capture data by:
1. Loading and filtering AMASS motion files
2. Converting coordinate systems and applying transformations
3. Creating both original and mirrored versions of motion sequences
4. Saving processed data in a standardized format

The script maintains compatibility with HumanML3D dataset structure.
"""

import sys, os
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.rotation_tools import matrot2aa, aa2matrot
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import pathlib
from collections import defaultdict
import pandas as pd
from collections import defaultdict
import os
import trimesh
from scipy.spatial.transform import Rotation as R
import smplx
import argparse


def parse_arguments():
    """Parse command line arguments for dataset processing configuration."""
    parser = argparse.ArgumentParser(description='Dataset processing options')
    parser.add_argument("--smplx_path", type=str, required=False, 
                       default="./model_files/smplx_models", 
                       help="Path to SMPL-X model files")
    parser.add_argument("--dataset_path_original", type=str, required=False, 
                       default="/data/datasets/AMASS_original_smplx", 
                       help="Path to original AMASS dataset")
    parser.add_argument("--dataset_path_processed", type=str, required=False, 
                       default="/data/datasets/AMASS", 
                       help="Path to processed AMASS dataset")
    parser.add_argument("--index_path", type=str, required=False, 
                       default="./preprocess/index.csv", 
                       help="Path to index file")
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug mode")
    parser.add_argument("--ex_fps", type=int, required=False, default=30, 
                       help="Target frame rate for processing")
    
    return parser.parse_args()


def initialize_smplx_model(smplx_model_path, num_shape_params=16):
    """
    Initialize SMPL-X body model for motion processing.
    
    Args:
        smplx_model_path (str): Path to SMPL-X model files
        num_shape_params (int): Number of shape parameters to use
        
    Returns:
        smplx.SMPLX: Initialized SMPL-X model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    neutral_body_model = smplx.create(
        smplx_model_path,
        model_type='smplx',
        gender='NEUTRAL_2020',
        use_face_contour=False,
        num_betas=num_shape_params,
        num_expression_coeffs=100,
        ext='npz',
        use_pca=False,
    ).eval().to(device)
    
    return neutral_body_model


def load_humanml3d_mapping(index_file_path):
    """
    Load mapping between AMASS source paths and HumanML3D names.
    
    Args:
        index_file_path (str): Path to the index CSV file
        
    Returns:
        dict: Mapping from source path to (new_name, start_frame, end_frame)
    """
    index_dataframe = pd.read_csv(index_file_path)
    total_sequences = index_dataframe.shape[0]
    
    humanml3d_mapping = {}
    
    for i in tqdm(range(total_sequences), desc="Loading HumanML3D mapping"):
        source_path = index_dataframe.loc[i]['source_path']
        
        # Apply path standardization for non-BMLhandball datasets
        if 'BMLhandball' not in source_path:
            # Standardize dataset directory names
            source_path = (source_path
                        .replace('/BioMotionLab_NTroje/', '/BMLrub/')
                        .replace('/DFaust_67/', '/DFaust/')
                        .replace('/MPI_mosh/', '/MoSh/')
                        .replace('/MPI_HDM05/', '/HDM05/')
                        .replace('/MPI_Limits/', '/PosePrior/')
                        .replace('/SSM_synced/', '/SSM/')
                        .replace('/TCD_handMocap/', '/TCDHands/')
                        .replace('/Transitions_mocap/', '/Transitions/')
                        .replace('.npy', '.npz')
                        .replace(' ', '_')
                        .replace('_poses', '_stageii'))
        
        # Create standardized file identifier
        source_path = source_path.replace('./pose_data/', '').replace('/', '_')
        new_name = index_dataframe.loc[i]['new_name']
        start_frame = index_dataframe.loc[i]['start_frame']
        end_frame = index_dataframe.loc[i]['end_frame']
        
        humanml3d_mapping[source_path] = (new_name, start_frame, end_frame)
    
    return humanml3d_mapping


def collect_and_filter_motion_files(amass_data_directory, humanml3d_mapping, debug_mode=False):
    """
    Collect and filter motion files based on various criteria.
    
    Args:
        amass_data_directory (pathlib.Path): Directory containing AMASS data
        humanml3d_mapping (dict): Mapping from source paths to HumanML3D names
        debug_mode (bool): Whether to run in debug mode
        
    Returns:
        list: List of valid motion files to process
    """
    # List of datasets to include in processing
    included_datasets = [
        'ACCAD', 'BMLrub', 'BMLhandball', 'BMLmovi', 'CMU', 'DFaust',
        'EKUT', 'Eyes_Japan_Dataset', 'HumanEva', 'KIT', 'HDM05',
        'PosePrior', 'MoSh', 'SFU', 'SSM', 'TCDHands', 'TotalCapture', 'Transitions'
    ]
    
    # Find all .npz files in the specified datasets
    all_motion_files = [
        pathlib.Path(f'{root}/{filename}')
        for root, dirs, files in os.walk(amass_data_directory)
        for filename in files
        if filename.endswith('.npz') and 
           any(dataset in pathlib.Path(root).parts for dataset in included_datasets)
    ]
    all_motion_files.sort()
    
    # Initialize tracking lists for file categorization
    valid_motion_files = []
    files_not_readable = []
    files_not_in_humanml3d = []
    files_incompatible_model = []
    files_missing_framerate = []
    files_invalid_framerate = []
    
    # Track statistics
    model_type_counts = defaultdict(int)
    framerate_counts = defaultdict(int)
    
    print(f"Processing {len(all_motion_files)} motion files...")
    
    # Process each file and categorize based on various criteria
    for motion_file in tqdm(all_motion_files, ncols=150, desc="Filtering motion files"):
        source_identifier = motion_file.relative_to(amass_data_directory).as_posix().replace('/', '_')
        
        # Check if file is in HumanML3D mapping
        if source_identifier not in humanml3d_mapping:
            files_not_in_humanml3d.append(motion_file)
            continue
        
        # Skip detailed checks in debug mode
        if debug_mode:
            valid_motion_files.append(motion_file)
            continue
        
        # Try to load and validate the motion file
        try:
            motion_data = np.load(motion_file, allow_pickle=True)
        except zipfile.BadZipFile:
            files_not_readable.append(motion_file)
            continue
        
        # Check model type compatibility
        model_type = motion_data['surface_model_type'].item()
        model_type_counts[model_type] += 1
        
        if model_type not in {'smplx', 'smplx_locked_head'}:
            files_incompatible_model.append(motion_file)
            continue
        
        # Check for frame rate information
        if 'mocap_frame_rate' not in motion_data:
            files_missing_framerate.append(motion_file)
            continue
        
        # Track frame rate statistics
        frame_rate = int(motion_data['mocap_frame_rate'].item())
        framerate_counts[frame_rate] += 1
        
        valid_motion_files.append(motion_file)
    
    # Print comprehensive statistics
    print(f'\nFile Processing Statistics:')
    print(f'Total files found: {len(all_motion_files)}')
    print(f'Files not in HumanML3D mapping: {len(files_not_in_humanml3d)}')
    print(f'Valid files for processing: {len(valid_motion_files)}')
    print(f'Files not readable: {len(files_not_readable)}')
    print(f'Files with incompatible model type: {len(files_incompatible_model)}')
    print(f'Files missing frame rate info: {len(files_missing_framerate)}')
    print(f'Files with invalid frame rate: {len(files_invalid_framerate)}')
    print(f'Model types found: {sorted(model_type_counts.items())}')
    print(f'Frame rates found: {sorted(framerate_counts.items())}')
    
    # Verify that all files are accounted for
    total_categorized = (len(valid_motion_files) + len(files_not_in_humanml3d) + 
                        len(files_not_readable) + len(files_incompatible_model) + 
                        len(files_missing_framerate) + len(files_invalid_framerate))
    assert len(all_motion_files) == total_categorized, "File count mismatch in categorization"
    
    return valid_motion_files


def save_mesh_as_obj(output_path, vertices, faces):
    """
    Save mesh vertices and faces as OBJ file.
    
    Args:
        output_path (str): Path to save the OBJ file
        vertices (np.ndarray): Mesh vertices
        faces (np.ndarray): Mesh faces
    """
    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    triangle_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=vertex_colors, process=False)
    triangle_mesh.export(output_path)


def create_rotation_matrix(rx, ry, rz, use_degrees=True):
    """
    Create rotation matrix from Euler angles.
    
    Args:
        rx, ry, rz (float): Rotation angles around x, y, z axes
        use_degrees (bool): Whether angles are in degrees
        
    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    if use_degrees:
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    
    sin_x, sin_y, sin_z = np.sin(rx), np.sin(ry), np.sin(rz)
    cos_x, cos_y, cos_z = np.cos(rx), np.cos(ry), np.cos(rz)

    # Create rotation matrices for each axis
    rotation_x = np.array([[1.0, 0.0, 0.0],
                          [0.0, cos_x, -sin_x],
                          [0.0, sin_x, cos_x]])
    
    rotation_y = np.array([[cos_y, 0.0, sin_y],
                          [0.0, 1.0, 0.0],
                          [-sin_y, 0.0, cos_y]])
    
    rotation_z = np.array([[cos_z, -sin_z, 0.0],
                          [sin_z, cos_z, 0.0],
                          [0.0, 0.0, 1.0]])

    # Combine rotations in ZYX order
    combined_rotation = np.matmul(np.matmul(rotation_z, rotation_y), rotation_x)
    return combined_rotation


def process_and_align_motion_sequence(motion_file_path, output_file_path, output_mirrored_path, 
                                    mesh_output_path, mesh_mirrored_path, start_frame, end_frame, 
                                    right_joint_indices, left_joint_indices, body_model, 
                                    target_fps, num_shape_params, coordinate_transform_matrix):
    """
    Process and align a single motion sequence from AMASS data.
    
    Args:
        motion_file_path (pathlib.Path): Input motion file path
        output_file_path (pathlib.Path): Output path for processed motion
        output_mirrored_path (pathlib.Path): Output path for mirrored motion
        mesh_output_path (pathlib.Path): Output path for mesh data
        mesh_mirrored_path (pathlib.Path): Output path for mirrored mesh data
        start_frame (int): Starting frame for sequence extraction
        end_frame (int): Ending frame for sequence extraction
        right_joint_indices (list): Indices of right-side joints
        left_joint_indices (list): Indices of left-side joints
        body_model: SMPL-X body model
        target_fps (int): Target frame rate for output
        num_shape_params (int): Number of shape parameters
        coordinate_transform_matrix (np.ndarray): Coordinate system transformation matrix
        
    Returns:
        int: Original frame rate of the motion file, or 0 if processing failed
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load motion data
    try:
        motion_data = np.load(motion_file_path, allow_pickle=True)
    except zipfile.BadZipFile:
        return 0
    
    # Extract frame rate information
    original_fps = 0
    try:
        original_fps = motion_data['mocap_frame_rate']
    except:
        return original_fps

    # Verify model compatibility
    assert motion_data['surface_model_type'].item() == 'smplx', "Only SMPL-X models are supported"

    # Calculate downsampling rate and apply it
    downsample_factor = int(original_fps / target_fps)
    pose_data = motion_data['poses'][::downsample_factor, ...]
    translation_data = motion_data['trans'][::downsample_factor, ...]

    # Adjust frame indices for new sampling rate
    adjusted_start_frame = int(start_frame * 1.5)  # 1.5 is the ratio of original fps to target fps
    adjusted_end_frame = int(end_frame * 1.5)

    original_fps = int(original_fps)
    
    # Apply dataset-specific temporal adjustments
    motion_file_str = motion_file_path.as_posix()
    if 'humanact12' not in motion_file_str:
        # Apply dataset-specific frame offsets
        if 'Eyes_Japan_Dataset' in motion_file_str:
            pose_data = pose_data[3 * target_fps:]
            translation_data = translation_data[3 * target_fps:]
        elif 'HDM05' in motion_file_str:
            pose_data = pose_data[3 * target_fps:]
            translation_data = translation_data[3 * target_fps:]
        elif 'TotalCapture' in motion_file_str:
            pose_data = pose_data[1 * target_fps:]
            translation_data = translation_data[1 * target_fps:]
        elif 'PosePrior' in motion_file_str:
            pose_data = pose_data[1 * target_fps:]
            translation_data = translation_data[1 * target_fps:]
        elif 'Transitions' in motion_file_str:
            pose_data = pose_data[int(0.5 * target_fps):]
            translation_data = translation_data[int(0.5 * target_fps):]

        # Extract the specified frame range
        pose_data = pose_data[adjusted_start_frame:adjusted_end_frame]
        translation_data = translation_data[adjusted_start_frame:adjusted_end_frame]

    # Calculate body model offset using neutral pose
    neutral_body_output = body_model(
        betas=torch.Tensor(0 * motion_data['betas'][:num_shape_params]).to(device).reshape(1, num_shape_params),
        transl=torch.Tensor(0 * translation_data[0:1]).to(device).reshape(1, 3),
        expression=torch.zeros([1, 100]).to(device),
        jaw_pose=torch.zeros([1, 3]).to(device),
        global_orient=torch.Tensor(0 * pose_data[0:1, :3]).to(device),
        body_pose=torch.Tensor(0 * pose_data[0:1, 3:21 * 3 + 3]).to(device),
        left_hand_pose=torch.Tensor(0 * pose_data[0:1, 25 * 3:40 * 3]).to(device),
        right_hand_pose=torch.Tensor(0 * pose_data[0:1, 40 * 3:55 * 3]).to(device),
        leye_pose=torch.zeros([1, 3]).to(device),
        reye_pose=torch.zeros([1, 3]).to(device),
    )
    body_offset = neutral_body_output.joints[0, 0].cpu().numpy()

    # Apply DFaust-specific height correction
    if "DFaust" in motion_file_str:
        translation_data[:, 2] += 0.62  # Manual correction for foot height offset

    # Process global orientation
    global_orientation = torch.Tensor(pose_data[:, :3])
    global_orientation = aa2matrot(global_orientation)
    global_orientation = torch.Tensor(coordinate_transform_matrix).unsqueeze(0) @ global_orientation

    # Calculate initial orientation for alignment
    first_frame_orientation = global_orientation[0].clone()

    # Calculate XZ plane direction for alignment
    xz_direction = np.array([first_frame_orientation[0, 2], 0, first_frame_orientation[2, 2]])
    xz_direction /= np.linalg.norm(xz_direction)
    
    # Create rotation matrix to align to target direction
    current_rotation_matrix = R.from_euler('y', np.arctan2(xz_direction[0], xz_direction[2])).as_matrix()
    target_rotation_matrix = np.eye(3)
    alignment_rotation_matrix = target_rotation_matrix @ current_rotation_matrix.T

    # Apply coordinate transformations
    alignment_rotation_tensor = torch.tensor(alignment_rotation_matrix).float()
    global_orientation = alignment_rotation_tensor @ global_orientation
    global_orientation = matrot2aa(global_orientation)
    global_orientation = global_orientation.numpy()
    
    # Update pose data with transformed global orientation
    pose_data[:, :3] = global_orientation
    
    # Transform translation data
    translation_data = np.dot(coordinate_transform_matrix, translation_data.T).T
    translation_data = np.dot(alignment_rotation_matrix, translation_data.T).T

    # Adjust translations for normalization
    translation_data[:, 1:] -= body_offset[1:]
    translation_data[:, 1:] += 0.1
    translation_data[:, 0] -= translation_data[0, 0]
    translation_data[:, 2] -= translation_data[0, 2]

    # Check if sequence has valid length
    if len(pose_data) == 0:
        return 0

    # Save processed motion data
    processed_motion_data = dict(motion_data)
    processed_motion_data['poses'] = pose_data
    processed_motion_data['trans'] = translation_data
    processed_motion_data['mocap_frame_rate'] = 30.0
    processed_motion_data['expressions'] = torch.zeros([pose_data.shape[0], 10], dtype=torch.float32)
    np.savez(output_file_path, **processed_motion_data)

    # Create mirrored version of the motion
    pose_data_reshaped = pose_data.reshape(-1, 55, 3)
    mirrored_pose_data = pose_data_reshaped.copy()
    mirrored_translation_data = translation_data.copy()
    
    # Swap left and right joints for mirroring
    mirrored_pose_data[:, left_joint_indices] = pose_data_reshaped[:, right_joint_indices]
    mirrored_pose_data[:, right_joint_indices] = pose_data_reshaped[:, left_joint_indices]
    
    # Mirror Y and Z rotations
    mirrored_pose_data[:, :, 1:3] *= -1
    
    # Mirror X translation
    mirrored_translation_data[..., 0] *= -1
    mirrored_translation_data[:, 0] -= mirrored_translation_data[0, 0]
    mirrored_translation_data[:, 2] -= mirrored_translation_data[0, 2]
    
    # Reshape back to original format
    mirrored_pose_data = mirrored_pose_data.reshape(-1, 55 * 3)

    # Save mirrored motion data
    mirrored_motion_data = dict(motion_data)
    mirrored_motion_data['poses'] = mirrored_pose_data
    mirrored_motion_data['trans'] = mirrored_translation_data
    mirrored_motion_data['mocap_frame_rate'] = 30.0
    mirrored_motion_data['expressions'] = torch.zeros([mirrored_pose_data.shape[0], 10], dtype=torch.float32)
    np.savez(output_mirrored_path, **mirrored_motion_data)

    return original_fps


def setup_joint_mapping(smplx_model_path):
    """
    Set up joint mapping for left-right mirroring operations.
    
    Args:
        smplx_model_path (str): Path to SMPL-X model files
        
    Returns:
        tuple: (left_joint_indices, right_joint_indices, joints_to_exclude)
    """
    model_params_file = pathlib.Path(os.path.join(smplx_model_path, 'smplx/SMPLX_NEUTRAL_2020.npz'))
    model_params = np.load(model_params_file, allow_pickle=True)
    joint_name_to_index = model_params['joint2num'].item()
    index_to_joint_name = {v: k for k, v in joint_name_to_index.items()}

    # Identify left and right joints for mirroring
    left_joint_indices = []
    right_joint_indices = []
    
    for joint_name in joint_name_to_index:
        if joint_name.startswith('L_'):
            left_joint_name = joint_name
            right_joint_name = joint_name.replace('L_', 'R_')
            left_joint_indices.append(joint_name_to_index[left_joint_name])
            right_joint_indices.append(joint_name_to_index[right_joint_name])

    # Define joints to be excluded from processing
    joints_to_exclude = [
        joint_name_to_index['Jaw'],
        joint_name_to_index['L_Eye'],
        joint_name_to_index['R_Eye']
    ]

    # Print joint mapping information
    print(f'\nJoint Mapping Information:')
    print(f'Number of joint pairs to swap: {len(left_joint_indices)}')
    print(f'Left joint indices: {left_joint_indices}')
    print(f'Right joint indices: {right_joint_indices}')
    print(f'Joints to exclude: {joints_to_exclude}')
    print('\nJoint pair mappings:')
    for left_idx, right_idx in sorted(zip(left_joint_indices, right_joint_indices)):
        left_name = index_to_joint_name[left_idx]
        right_name = index_to_joint_name[right_idx]
        print(f'{left_name:10} ({left_idx:2}) <--> ({right_idx:2}) {right_name}')

    return left_joint_indices, right_joint_indices, joints_to_exclude


def main():
    """Main function to orchestrate the AMASS dataset processing pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    debug_mode = args.debug
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_fps = args.ex_fps
    num_shape_params = 16  # Number of shape parameters for SMPL-X model

    print("Starting AMASS dataset processing...")
    print(f"Using device: {device}")
    print(f"Target FPS: {target_fps}")
    print(f"Debug mode: {debug_mode}")

    # Initialize SMPL-X model
    print("\nInitializing SMPL-X model...")
    neutral_body_model = initialize_smplx_model(args.smplx_path, num_shape_params)

    # Load HumanML3D mapping
    print("\nLoading HumanML3D mapping...")
    humanml3d_mapping = load_humanml3d_mapping(args.index_path)
    print(f"Loaded {len(humanml3d_mapping)} motion sequence mappings")

    # Collect and filter motion files
    print("\nCollecting and filtering motion files...")
    amass_data_directory = pathlib.Path(args.dataset_path_original)
    valid_motion_files = collect_and_filter_motion_files(
        amass_data_directory, humanml3d_mapping, debug_mode
    )

    # Set up joint mapping for mirroring
    print("\nSetting up joint mapping...")
    left_joint_indices, right_joint_indices, joints_to_exclude = setup_joint_mapping(args.smplx_path)

    # Define coordinate transformation matrix for alignment
    coordinate_transform_matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0]
    ])

    # Define output directories
    processed_data_directory = pathlib.Path(os.path.join(args.dataset_path_processed, 'amass_data_align'))
    mesh_output_directory = pathlib.Path(os.path.join(args.dataset_path_processed, 'mesh_save'))
    mesh_mirrored_directory = pathlib.Path(os.path.join(args.dataset_path_processed, 'mesh_save_mirror'))

    # Process each motion file
    print(f"\nProcessing {len(valid_motion_files)} motion files...")
    
    for motion_file in tqdm(valid_motion_files, desc='Processing AMASS sequences', ncols=150):
        # Get mapping information for this file
        source_identifier = motion_file.relative_to(amass_data_directory).as_posix().replace('/', '_')
        new_name, start_frame, end_frame = humanml3d_mapping[source_identifier]
        
        # Define output file paths
        output_file_path = processed_data_directory / new_name.replace('.npy', '.npz')
        output_mirrored_path = processed_data_directory / ('M' + new_name.replace('.npy', '.npz'))
        mesh_output_path = mesh_output_directory / new_name
        mesh_mirrored_path = mesh_mirrored_directory / ('M' + new_name)

        # Create output directories if they don't exist
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Process and align motion data
        process_and_align_motion_sequence(
            motion_file, output_file_path, output_mirrored_path,
            mesh_output_path, mesh_mirrored_path, start_frame, end_frame,
            right_joint_indices, left_joint_indices, neutral_body_model,
            target_fps, num_shape_params, coordinate_transform_matrix
        )

    print(f"\nProcessing completed!")
    print(f"Processed data saved to: {args.dataset_path_processed}")


if __name__ == "__main__":
    main()


