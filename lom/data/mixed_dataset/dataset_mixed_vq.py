import random
import pickle
import os
import numpy as np
import codecs as cs
import torch
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
import pandas as pd
import math
import textgrid as tg
import h5py
from .utils.split_transcript import split_and_merge_sentences
import smplx
from tqdm import tqdm
from .data_tools import joints_list
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HAND,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HAND,
    BEAT_SMPLX_LOWER
)
from lom.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np

class MixedDatasetVQ(data.Dataset):
    def __init__(
        self,
        args,
        dataset_configs,
        split,
        # unit_length=4,
        # fps=20,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        task_path=None,
        mean=None,
        std=None,
        motion_representation="rotation",
        smpl_path=None,
        njoints=55,
        use_cache=False,  # Whether to load data from cache when available
        save_cache=False,  # Whether to save processed data to cache
        cache_format="pkl", # Format to use for caching: "h5", "npz", or "pkl"
        **kwargs,
    ):
        """
        Initializes the dataset class.

        Parameters:
        - dataset_configs: List of configurations for different datasets.
        - split: Specifies the data split (train/val/test).
        - args: Additional arguments.
        - unit_length: Length of the units for data processing.
        - fps: Frames per second for motion data.
        - tiny: Whether to use a small subset for debugging.
        - debug: If True, enables debug mode.
        - stage: Specifies the training stage.
        - task_path: Path to the task instructions file.
        - motion_representation: Specifies the motion representation.
        - use_cache: Whether to load data from cache when available.
        - save_cache: Whether to save processed data to cache.
        - cache_format: Format to use for caching ("h5", "npz", or "pkl").
        """
        # Store debug flag for cache path generation
        self.debug = debug
        
        # Set max data size depending on debug mode
        if tiny or debug:
            self.maxdata = 10
        else:
            self.maxdata = 1e10

        self.args = args
        self.dataset_configs = dataset_configs
        self.task_path = task_path
        # self.unit_length = unit_length
        self.stage = stage
        self.split = split
        self.njoints = njoints
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format
        self.smpl_path = smpl_path  # Store the path for later use if needed
        self.motion_representation = motion_representation

        # Store kwargs for SMPLX initialization if needed
        if motion_representation == "rotation":
            # self.select_type = kwargs['cfg']['Selected_type']
            # self.select_part = kwargs['cfg']["Representation_type"].get(self.select_type)
            # Initialize the joint masks for different body parts
            self.joint_mask_upper = JOINT_MASK_UPPER
            self.joint_mask_lower = JOINT_MASK_LOWER
            self.joint_mask_hand = JOINT_MASK_HAND
            self.joint_mask_face = JOINT_MASK_FACE
            self.joint_mask_full = JOINT_MASK_FULL
            # We'll initialize SMPLX only when needed, not upfront
            self.smplx_2020 = None
        elif motion_representation == "h3d":
            self.h3d_mean = mean
            self.h3d_std = std

        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []
        # Load each dataset based on its type from the configuration
        for config in dataset_configs:
            # dataset_name = config.get("type")
            dataset_name = config.get("name")
            if dataset_name == "amass_h3d":
                self._load_amass_h3d(config)
                self.data_dict.update(self.data_dict_amass_h3d)
                self.metadata.extend(self.metadata_amass_h3d)
            elif dataset_name == "AMASS":
                self._load_amass(config)
                self.data_dict.update(self.data_dict_amass)
                self.metadata.extend(self.metadata_amass)
            elif dataset_name == "BEAT2":
                # if self.split == 'test':
                #     continue
                self._load_beat2(config)
                self.data_dict.update(self.data_dict_beat2)
                self.metadata.extend(self.metadata_beat2)
            else:
                raise NotImplementedError(f"Unknown dataset name {dataset_name}")

        print(len(self.metadata))

    def _get_cache_path(self, dataset_path, dataset_type):
        """
        Generate a cache file path based on dataset path and type.
        
        Parameters:
        - dataset_path: Path to the original dataset
        - dataset_type: Type of the dataset (e.g., 'beat2', 'amass')
        
        Returns:
        - Path to the cache file
        """
        # Use the dataset's own directory for caching
        cache_dir = os.path.join(dataset_path, 'cache')
        
        # Generate a unique filename based on dataset configuration
        # Include debug identifier if in debug mode
        config_str = f"{dataset_type}_{self.split}_vq"
        if self.debug:
            config_str = f"{dataset_type}_{self.split}_vq_debug"
        
        # Set the file extension based on the cache format
        if self.cache_format == "h5":
            ext = ".h5"
        elif self.cache_format == "npz":
            ext = ".npz"
        else:  # Default to pickle
            ext = ".pkl"
            
        cache_path = os.path.join(cache_dir, f"{config_str}{ext}")
        return cache_path

    def _save_to_cache(self, cache_path, data_dict, metadata, dataset_name):
        """
        Save processed data to cache.
        
        Parameters:
        - cache_path: Path where to save the cache file
        - data_dict: Dictionary containing processed data (NumPy arrays)
        - metadata: List of metadata entries
        - dataset_name: Type of the dataset (for logging)
        """
        if not self.save_cache:
            return
            
        try:
            # Create cache directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            
            print(f"Saving {dataset_name} dataset to cache: {cache_path}")
            
            # Convert any remaining PyTorch tensors to NumPy arrays for consistency
            numpy_data_dict = {}
            for key, item in data_dict.items():
                numpy_item = {}
                for attr_key, attr_value in item.items():
                    if isinstance(attr_value, torch.Tensor):
                        # Convert torch tensors to numpy arrays
                        numpy_item[attr_key] = attr_value.cpu().numpy()
                    else:
                        numpy_item[attr_key] = attr_value
                numpy_data_dict[key] = numpy_item
            
            # Save based on the selected format
            if self.cache_format == "pkl":
                # Simple pickle save - fastest and most compatible
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'data_dict': numpy_data_dict,
                        'metadata': metadata
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            elif self.cache_format == "npz":
                # Faster uncompressed saving with NumPy
                np.savez(cache_path, 
                    data_dict=numpy_data_dict, 
                    metadata=metadata
                )
                
            elif self.cache_format == "h5":
                # HDF5 format - good for large datasets with compression
                with h5py.File(cache_path, 'w') as f:
                    # Save metadata as a JSON string
                    metadata_str = json.dumps(metadata)
                    # Convert metadata to a list of bytes to ensure it's not treated as a scalar
                    metadata_bytes = np.frombuffer(metadata_str.encode('utf-8'), dtype='uint8')
                    # Now we can safely use compression since it's definitely an array
                    f.create_dataset('metadata', data=metadata_bytes, compression='gzip')
                    
                    # Create a group for data_dict
                    data_group = f.create_group('data_dict')
                    
                    # Save each entry in numpy_data_dict
                    for key, item in numpy_data_dict.items():
                        # Create a group for each item (entry)
                        item_group = data_group.create_group(key)
                        
                        # Save each attribute of the item
                        for attr_key, attr_value in item.items():
                            if attr_value is None:
                                # Store None as a special value
                                item_group.attrs[attr_key] = 'None'
                            elif isinstance(attr_value, (str, int, float, bool)):
                                # Store simple types as attributes
                                item_group.attrs[attr_key] = attr_value
                            elif isinstance(attr_value, dict):
                                # Store dictionaries as JSON-encoded strings
                                attr_group = item_group.create_group(attr_key)
                                for k, v in attr_value.items():
                                    if isinstance(v, (str, int, float, bool, list)):
                                        attr_group.attrs[k] = json.dumps(v)
                            elif isinstance(attr_value, np.ndarray):
                                # Only apply compression to non-scalar arrays
                                if attr_value.size > 1:
                                    item_group.create_dataset(
                                        attr_key, 
                                        data=attr_value,
                                        compression='gzip', 
                                        compression_opts=4
                                    )
                                else:
                                    # For scalar arrays, store without compression
                                    item_group.create_dataset(attr_key, data=attr_value)
                            elif attr_key == 'tasks' or attr_key == 'emotion_label':
                                # Handle special cases like tasks
                                item_group.attrs[attr_key] = json.dumps(attr_value)
                            else:
                                # Try to convert other types to strings
                                try:
                                    item_group.attrs[attr_key] = str(attr_value)
                                except:
                                    print(f"Warning: Could not save attribute {attr_key} of type {type(attr_value)}")
                
            print(f"Successfully saved {dataset_name} cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving {dataset_name} cache: {str(e)}")
            # Print more details about the error for debugging
            import traceback
            traceback.print_exc()

    def _load_from_cache(self, cache_path, dataset_name):
        """
        Load processed data from cache.
        
        Parameters:
        - cache_path: Path to the cache file
        - dataset_name: Type of the dataset (for logging)
        
        Returns:
        - tuple (data_dict, metadata) if successful, (None, None) otherwise
        """

        
        if not self.use_cache:
            return None, None
        
        if not os.path.exists(cache_path):
            print(f"Cache file not found: {cache_path}")
            return None, None
        
        try:
            print(f"Loading {dataset_name} dataset from cache: {cache_path}")

            if self.cache_format == "pkl":
                # Simple pickle load - fastest method
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    data_dict = cached_data['data_dict']
                    metadata = cached_data['metadata']
                    
            elif self.cache_format == "npz":
                # Load from NPZ format - fast with NumPy
                loaded = np.load(cache_path, allow_pickle=True)
                
                # More robust metadata loading - handle both direct arrays and object arrays
                if loaded['metadata'].size == 1:
                    # Try to extract as a single object
                    metadata = loaded['metadata'].item()
                else:
                    # Load as a direct array (e.g., list of strings)
                    metadata = loaded['metadata']
                
                # More robust data_dict loading
                if loaded['data_dict'].size == 1:
                    # Normal case - data_dict is stored as an object array
                    data_dict = loaded['data_dict'].item()
                else:
                    # Handle edge case where it might be stored differently
                    print(f"Warning: Unexpected data_dict structure in {cache_path}")
                    data_dict = {}
            
            elif self.cache_format == "h5":
                # Load from HDF5 format - good for large datasets
                data_dict = {}
                with h5py.File(cache_path, 'r') as f:
                    # Load metadata
                    try:
                        # Try to load metadata as a byte array
                        metadata_bytes = f['metadata'][()]
                        metadata_str = metadata_bytes.tobytes().decode('utf-8')
                        metadata = json.loads(metadata_str)
                    except:
                        # Fallback: try loading directly as a string attribute
                        try:
                            metadata_str = f.attrs.get('metadata', '[]')
                            metadata = json.loads(metadata_str)
                        except:
                            # Last resort: just create an empty list
                            print(f"Warning: Could not load metadata from {cache_path}")
                            metadata = []
                    
                    # Load data_dict
                    data_group = f['data_dict']
                    for key in data_group.keys():
                        item_group = data_group[key]
                        item = {}
                        
                        # Load each attribute
                        for attr_key in item_group.keys():
                            try:
                                attr_value = item_group[attr_key][()]
                                item[attr_key] = attr_value
                            except:
                                print(f"Warning: Could not load attribute {attr_key} from {cache_path}")
                        
                        # Load attributes stored as metadata
                        for attr_key in item_group.attrs.keys():
                            attr_value = item_group.attrs[attr_key]
                            if attr_value == 'None':
                                item[attr_key] = None
                            elif attr_key in ['tasks', 'emotion_label'] or isinstance(attr_value, str) and attr_value.startswith('{') and attr_value.endswith('}'):
                                # Try to parse as JSON
                                try:
                                    item[attr_key] = json.loads(attr_value)
                                except:
                                    item[attr_key] = attr_value
                            else:
                                item[attr_key] = attr_value
                        
                        data_dict[key] = item
            
            print(f"Successfully loaded {dataset_name} cache from {cache_path}")
            return data_dict, metadata
            
        except Exception as e:
            print(f"Error loading {dataset_name} cache: {str(e)}")
            # Print more details about the error for debugging
            import traceback
            traceback.print_exc()
            return None, None

    def _initialize_smplx_if_needed(self):
        """
        Initialize the SMPLX model if it hasn't been initialized yet.
        This is done lazily to avoid unnecessary GPU memory usage when loading from cache.
        """
        if self.smplx_2020 is None and (self.motion_representation == "rotation"):
            print("Initializing SMPLX model for data processing...")
            self.smplx_2020 = smplx.create(self.smpl_path,
                model_type='smplx',
                gender='NEUTRAL_2020',
                use_face_contour=False,
                num_betas=300,
                num_expression_coeffs=100,
                ext='npz',
                use_pca=False,
                ).cuda().eval()

    def _load_beat2(self, config):
        """
        Load the BEAT2 dataset based on the configuration.

        Parameters:
            config: Configuration dictionary for BEAT2.
        """
        # Get cache path for this specific dataset
        data_root = self.args["BEAT2"].ROOT
        cache_path = self._get_cache_path(data_root, "BEAT2")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "BEAT2")
        if data_dict is not None:
            self.data_dict_beat2 = data_dict
            self.metadata_beat2 = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing BEAT2 dataset...")
        
        # Set up dataset parameters
        self.data_root_beat2 = data_root
        self.ori_length = config.pose_length
        additional_data = config.additional_data
        training_speakers = config.training_speakers
        pose_rep = config.pose_rep
        pose_fps_beat2 = config.pose_fps
        
        # Load split rules from CSV file
        split_rule = pd.read_csv(pjoin(data_root, "train_test_split.csv"))
        
        # Filter files based on training speakers and split type
        if self.split == 'token':
            # For token split, only filter by training speakers
            self.selected_file = split_rule.loc[
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
        else:
            # For other splits, filter by both split type and training speakers
            self.selected_file = split_rule.loc[
                (split_rule['type'] == self.split) &
                ((split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers)))
            ]
            # Include additional data if specified
            if additional_data:
                split_b = split_rule.loc[
                    (split_rule['type'] == 'additional') & 
                    (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))
                ]
                self.selected_file = pd.concat([self.selected_file, split_b])

        self.data_dict_beat2 = {}
        self.metadata_beat2 = []

        # Process each file in the selected files
        for index, file_name in tqdm(self.selected_file.iterrows()):
            f_name = file_name["id"]
            pose_file = pjoin(self.data_root_beat2, pose_rep, f_name + ".npz")
            
            # try:
                # Load pose data from NPZ file
            pose_data = np.load(pose_file, allow_pickle=True)
            poses = pose_data["poses"]
            n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
            trans = pose_data["trans"]
            betas = pose_data["betas"]
            # Repeat betas for each frame
            betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)

            expressions = pose_data["expressions"]

            # Apply joint mask to filter relevant joints
            pose_processed = poses * JOINT_MASK_FULL
            pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
            
            if self.motion_representation == 'rotation':
                # Calculate foot contacts using existing function or load from cache
                foot_contacts_path = pjoin(self.data_root_beat2, 'foot_contacts', f_name + '.npy')
                if os.path.exists(foot_contacts_path):
                    contacts = np.load(foot_contacts_path)
                else:
                    contacts = self.comput_foot_contacts(pose_data)
                    os.makedirs(pjoin(self.data_root_beat2, 'foot_contacts'), exist_ok=True)
                    np.save(foot_contacts_path, contacts)
                # Concatenate foot contacts to pose data
                pose_processed = np.concatenate([pose_processed, contacts], axis=1)

            # Extract different components from processed pose
            tar_pose = pose_processed[ :, :165]  # Body pose (55 joints * 3)
            tar_contact = pose_processed[ :, 165:169]  # Foot contacts (4 values)
            tar_exps = expressions  # Facial expressions
            tar_trans = trans  # Translation

            # Extract and convert jaw pose data
            tar_pose_jaw = tar_pose[:, 66:69]  # Jaw pose (3D rotation)
            tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)  # Convert to 6D representation
            
            # Concatenate jaw pose and expressions for face data
            tar_pose_face = np.concatenate([tar_pose_jaw_6d, tar_exps], axis=1)

            # Extract and convert hand pose data
            tar_pose_hands = tar_pose[:, 25 * 3:55 * 3].reshape(n, 30, 3)  # 30 hand joints
            tar_pose_hands_6d = axis_angle_to_6d_np(tar_pose_hands).reshape(n, 30 * 6)

            # Extract and convert upper body pose data
            tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)  # 13 upper body joints
            tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

            # Extract and convert lower body pose data
            tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)  # 9 lower body joints
            tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
            
            # Combine lower body pose with translation and contacts
            tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
            
            # Convert full pose to 6D representation
            tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55 * 6)

            # Calculate time segments
            round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_beat2
            if round_seconds_skeleton == 0:
                round_seconds_skeleton = 1
            clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0  # Start and end time in seconds
            clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_beat2, clip_e_t * pose_fps_beat2  # Start and end frame

            # Determine stride and cut length based on split type
            if self.split == 'test' or self.split == 'token':  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                stride = cut_length
                self.max_length = cut_length
            else:
                stride = int(config.stride)
                cut_length = int(self.ori_length)
            
            # Calculate number of subdivisions
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

            # Create segments of motion data
            for i in range(num_subdivision):  # cut into segments (e.g., 2s clips)

                start_idx = clip_s_f_pose + i * config.stride
                fin_idx = start_idx + cut_length
                
                # Extract segment from each data type
                sample_pose = tar_pose_6d[start_idx:fin_idx]
                sample_face = tar_pose_face[start_idx:fin_idx]
                sample_hand = tar_pose_hands_6d[start_idx:fin_idx]
                sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                sample_lower = tar_pose_lower[start_idx:fin_idx]
                sample_trans = tar_trans[start_idx:fin_idx]
                sample_shape = betas[start_idx:fin_idx]
                sample_expressions = expressions[start_idx:fin_idx]

                # Create unique name for this segment
                new_name = 'beat2_' + '%s_%d' % (f_name,i)
                
                # Store processed data
                self.data_dict_beat2[new_name] = {
                    'face': sample_face,
                    'hand': sample_hand,
                    'upper': sample_upper,
                    'lower': sample_lower,
                    'pose': sample_pose,
                    'shape': sample_shape,
                    'trans': sample_trans,
                    'exps': sample_expressions,
                    'id': f_name,
                    'dataset_name': 'beat2',
                }
                self.metadata_beat2.append(new_name)
                
            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2")

    def _load_amass_h3d(self, config):
        """
        Load the AMASS dataset in HumanML3D format based on the configuration.
        
        Parameters:
        - config: Configuration dictionary for AMASS_H3D.
        """
        # Get cache path for this specific dataset
        data_root = config.get("data_root")
        cache_path = self._get_cache_path(data_root, "amass_h3d")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "amass_h3d")
        if data_dict is not None:
            self.data_dict_amass_h3d = data_dict
            self.metadata_amass_h3d = metadata
            return
        
        print(f"Processing AMASS_H3D dataset...")
        
        # Set up dataset parameters
        self.data_root_amass_h3d = data_root
        pose_fps_amass = config.pose_fps
        motion_unit = config.motion_unit
        
        # Load split file containing dataset splits
        split_file = pd.read_csv(pjoin(self.data_root_amass_h3d, f"new_{self.split}.csv"))
        
        # Calculate maximum lengths for data
        self.max_length = int(config.pose_length)
        self.ori_length = config.pose_length

        self.metadata_amass_h3d = []
        self.data_dict_amass_h3d = {}
        
        ##################  AMASS in Humanml3d Format ##################
        # Process each file in the split
        for index, file_name in tqdm(split_file.iterrows()):
            f_name = file_name["id"]
            pose_file = pjoin(self.data_root_amass_h3d, 'new_joint_vecs', f_name+'.npy')
            
            # Load pre-processed joint vectors
            pose_each_file = np.load(pose_file, allow_pickle=True)
            
            # Normalize using pre-computed mean and std
            pose_each_file = (pose_each_file - self.h3d_mean) / self.h3d_std

            # Calculate time segments
            round_seconds_skeleton = pose_each_file.shape[0] // motion_unit
            if round_seconds_skeleton == 0:
                round_seconds_skeleton = 1
            clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0  # Start and end time
            clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * motion_unit  # Start and end frame

            # Determine stride and cut length based on split type
            if self.split == 'test':  # stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                stride = cut_length
                self.max_length = cut_length
            else:
                stride = int(config.stride)
                cut_length = int(int(self.ori_length))
            
            # Calculate number of subdivisions
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

            # Create segments of motion data
            for i in range(num_subdivision):  # cut into segments (e.g., 2s clips)
                start_idx = clip_s_f_pose + i * config.stride
                fin_idx = start_idx + cut_length
                
                # Extract segment
                sample_pose = pose_each_file[start_idx:fin_idx]
                
                # For H3D format, trans and shape are not used (set to zeros)
                sample_trans = np.zeros(1)
                sample_shape = np.zeros(1)
                
                # Only add valid segments (non-empty and non-zero)
                if sample_pose.any() and sample_pose.size != 0:
                    # Create unique name for this segment
                    new_name = 'amass_' + '%s_%d' % (f_name,i)
                    
                    # Store processed data
                    self.data_dict_amass_h3d[new_name] = {
                        'id' : f_name,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                    }
                    self.metadata_amass_h3d.append(new_name)
                    
            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass_h3d, self.metadata_amass_h3d, "amass_h3d")

    def _load_amass(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        # Get cache path for this specific dataset
        data_root = self.args["AMASS"].ROOT
        cache_path = self._get_cache_path(data_root, "AMASS")
        
        # Check if we should load from cache
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS")
        if data_dict is not None:
            self.data_dict_amass = data_dict
            self.metadata_amass = metadata
            return
        
        # We need to process data from scratch, so initialize SMPLX if needed
        self._initialize_smplx_if_needed()
        
        print(f"Processing AMASS dataset...")
        
        # Set up dataset parameters
        self.data_root_amass = data_root
        pose_fps_amass = config.pose_fps
        
        # Load train and test splits from text files
        split_file_train = pjoin(self.data_root_amass, 'train.txt')
        
        # Load training data IDs
        id_list_train = []
        with cs.open(split_file_train, "r") as f:
            for line in f.readlines():
                id_list_train.append(line.strip())

        split_file_test = pjoin(self.data_root_amass, 'test.txt')
        
        # Load test data IDs
        id_list_test = []
        with cs.open(split_file_test, "r") as f:
            for line in f.readlines():
                id_list_test.append(line.strip())

        # Select appropriate ID list based on split
        if self.split == 'train':
            id_list_amass = id_list_train
        elif self.split == 'test':
            id_list_amass = id_list_test
        else:
            # For other splits, use both train and test
            id_list_amass = id_list_train + id_list_test

        self.ori_length = config.pose_length
        # Calculate maximum lengths for data
        # self.max_length = int(config.pose_length)

        self.metadata_amass = []
        self.data_dict_amass = {}

        ##################  AMASS  ##################
        # Process each file
        for index, file_name in tqdm(enumerate(id_list_amass)):
            try:
                # Load pose data from aligned AMASS data
                pose_file = pjoin(self.data_root_amass, 'amass_data_align', file_name+'.npz')
                pose_data = np.load(pose_file, allow_pickle=True)
                
                # Calculate stride for downsampling to target FPS
                stride = int(30 / pose_fps_amass)

                # Extract and downsample pose data
                poses = pose_data["poses"][::stride]
                n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
                tar_trans = pose_data["trans"][::stride]

                # Pad betas to 300 dimensions (AMASS uses 16, but we need 300 for SMPLX)
                padded_betas = np.zeros(300)
                padded_betas[:16] = pose_data["betas"]
                betas = np.repeat(padded_betas.reshape(1, 300), n, axis=0)
                
                # Apply joint mask to filter relevant joints
                pose_processed = poses * JOINT_MASK_FULL
                pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
                
                # if self.select_type == 'full_rot' or self.select_type == 'separate_rot':
                if self.motion_representation == 'rotation':
                    # Calculate foot contacts for rotation representation
                    foot_contacts_path = pjoin(self.data_root_amass, 'foot_contacts', file_name + '.npy')
                    if os.path.exists(foot_contacts_path):
                        contacts = np.load(foot_contacts_path)
                    else:
                        contacts = self.comput_foot_contacts(pose_data)
                        os.makedirs(pjoin(self.data_root_amass, 'foot_contacts'), exist_ok=True)
                        np.save(foot_contacts_path, contacts)
                    # Concatenate foot contacts to pose data
                    pose_processed = np.concatenate([pose_processed, contacts], axis=1)
                
                # Extract body pose (55 joints * 3)
                tar_pose = pose_processed[:, :165]

                # Get foot contacts
                tar_contact = contacts

                # AMASS doesn't have facial expressions or hand poses, so we create zeros
                tar_pose_face = np.zeros((n, 106))  # No jaw pose or expressions
                tar_pose_hands_6d = np.zeros((n, 180))  # No hand pose
                
                # Extract and convert upper body pose data
                tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
                tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

                # Extract and convert lower body pose data
                tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
                tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
                
                # Combine lower body pose with translation and contacts
                tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
                
                # Convert full pose to 6D representation
                tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55 * 6)

                # Calculate time segments
                round_seconds_skeleton = tar_pose_6d.shape[0] // pose_fps_amass
                if round_seconds_skeleton == 0:
                    round_seconds_skeleton = 1
                clip_s_t, clip_e_t = 0, round_seconds_skeleton - 0
                clip_s_f_pose, clip_e_f_pose = clip_s_t * pose_fps_amass, clip_e_t * pose_fps_amass
            
                # Determine stride and cut length based on split type
                if self.split == 'test' or self.split == 'token':  # stride = length for test
                    cut_length = clip_e_f_pose - clip_s_f_pose
                    stride = cut_length
                    # self.max_length = cut_length
                else:
                    stride = int(config.stride)
                    cut_length = int(self.ori_length)
                
                # Calculate number of subdivisions
                num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / stride) + 1

                # Create segments of motion data
                for i in range(num_subdivision):
                    start_idx = clip_s_f_pose + i * stride
                    fin_idx = start_idx + cut_length
                    
                    # Skip if segment goes out of bounds
                    if fin_idx > tar_pose_6d.shape[0]:
                        continue
                    
                    # Extract segment from each data type    
                    sample_pose = tar_pose_6d[start_idx:fin_idx]
                    sample_face = tar_pose_face[start_idx:fin_idx]  # Empty for AMASS
                    sample_hand = tar_pose_hands_6d[start_idx:fin_idx]  # Empty for AMASS
                    sample_upper = tar_pose_upper_6d[start_idx:fin_idx]
                    sample_lower = tar_pose_lower[start_idx:fin_idx]
                    sample_trans = tar_trans[start_idx:fin_idx]
                    sample_shape = betas[start_idx:fin_idx]

                    # Create unique name for this segment
                    new_name = 'amass_' + '%s_%d' % (file_name, i)

                    # Store processed data
                    self.data_dict_amass[new_name] = {
                        'face': sample_face,
                        'hand': sample_hand,
                        'upper': sample_upper,
                        'lower': sample_lower,
                        'pose': sample_pose,
                        'shape': sample_shape,
                        'trans': sample_trans,
                        'id': file_name,
                        'dataset_name': 'amass',
                    }
                    self.metadata_amass.append(new_name)
            except Exception as e:
                # Skip files that can't be processed
                # print(f"Error processing file {f_name}: {str(e)}")
                continue

            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_amass, self.metadata_amass, "AMASS")

    def comput_foot_contacts(self, m_data):
        """
        Compute foot contacts from motion data.
        This method requires SMPLX, so we ensure it's initialized.
        
        Parameters:
        - m_data: Motion data dictionary containing betas, poses, trans, and expressions
        
        Returns:
        - contacts: Binary array indicating foot contacts (1 for contact, 0 for no contact)
        """
        # Make sure SMPLX is initialized
        self._initialize_smplx_if_needed()
        
        # Extract motion components
        betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
        n, c = poses.shape[0], poses.shape[1]  # n: number of frames, c: pose dimension
        
        # Determine the dimension of betas
        beta_dim = betas.shape[-1]  # get the last dimension of betas

        if beta_dim == 16:
            # AMASS dataset has 16 beta parameters, need to pad to 300
            padded_betas = np.zeros(300)
            padded_betas[:16] = betas
            exps = torch.zeros([n, 100], dtype=torch.float32).cuda()  # AMASS dataset has no expressions
        else:
            # BEAT2 dataset already has 300 beta parameters
            padded_betas = betas
            exps = torch.from_numpy(m_data["expressions"]).cuda().float()  # BEAT2 dataset has expressions

        # Prepare betas for all frames
        betas = padded_betas.reshape(1, 300)  # can be 16 or 300
        betas = np.tile(betas, (n, 1))

        # Convert all data to tensors on GPU
        betas = torch.from_numpy(betas).cuda().float()
        poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
        trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
        
        # Process in batches to avoid memory issues
        max_length = 128
        s, r = n // max_length, n % max_length  # s: number of full batches, r: remainder
        all_tensor = []
        
        # Process full batches
        for i in range(s):
            with torch.no_grad():
                # Run SMPLX forward pass to get joint positions
                joints = self.smplx_2020(
                    betas=betas[i * max_length:(i + 1) * max_length],
                    transl=trans[i * max_length:(i + 1) * max_length],
                    expression=exps[i * max_length:(i + 1) * max_length],
                    jaw_pose=poses[i * max_length:(i + 1) * max_length, 66:69],
                    global_orient=poses[i * max_length:(i + 1) * max_length, :3],
                    body_pose=poses[i * max_length:(i + 1) * max_length, 3:21 * 3 + 3],
                    left_hand_pose=poses[i * max_length:(i + 1) * max_length, 25 * 3:40 * 3],
                    right_hand_pose=poses[i * max_length:(i + 1) * max_length, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=poses[i * max_length:(i + 1) * max_length, 69:72],
                    reye_pose=poses[i * max_length:(i + 1) * max_length, 72:75],
                )['joints'][:, (7, 8, 10, 11), :].reshape(max_length, 4, 3).cpu()  # Extract foot joints
            all_tensor.append(joints)
            
        # Process remainder frames if any
        if r != 0:
            with torch.no_grad():
                joints = self.smplx_2020(
                    betas=betas[s * max_length:s * max_length + r],
                    transl=trans[s * max_length:s * max_length + r],
                    expression=exps[s * max_length:s * max_length + r],
                    jaw_pose=poses[s * max_length:s * max_length + r, 66:69],
                    global_orient=poses[s * max_length:s * max_length + r, :3],
                    body_pose=poses[s * max_length:s * max_length + r, 3:21 * 3 + 3],
                    left_hand_pose=poses[s * max_length:s * max_length + r, 25 * 3:40 * 3],
                    right_hand_pose=poses[s * max_length:s * max_length + r, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=poses[s * max_length:s * max_length + r, 69:72],
                    reye_pose=poses[s * max_length:s * max_length + r, 72:75],
                )['joints'][:, (7, 8, 10, 11), :].reshape(r, 4, 3).cpu()
            all_tensor.append(joints)
            
        # Concatenate all batches
        joints = torch.cat(all_tensor, axis=0)  # all, 4, 3
        
        # Calculate foot velocities
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)  # 4, all, 3
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        
        # Determine contacts based on velocity threshold
        contacts = (feetv < 0.01).numpy().astype(float)  # Contact when velocity < 0.01
        contacts = contacts.transpose(1, 0)  # all, 4

        return contacts

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset based on the given index.
        Converts NumPy arrays to PyTorch tensors for model usage.

        Parameters:
        - item: Index of the sample to retrieve.

        Returns:
        - A dictionary containing data with tensors for model input.
        """
        # Get the data key from metadata
        dataset_name = self.metadata[item]
        data = self.data_dict[dataset_name]
        
        # Create a copy of the data dictionary to avoid modifying the original
        formatted_data = {}
        
        # Convert NumPy arrays to PyTorch tensors for model usage
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                # Convert NumPy arrays to tensors with float32 dtype
                formatted_data[key] = torch.from_numpy(value).float()
            else:
                # Keep other data types as is (strings, etc.)
                formatted_data[key] = value
                
        # Get motion length from the pose data
        motion_len = formatted_data['pose'].shape[0] if 'pose' in formatted_data else 0
        
        # Add additional information needed for training
        formatted_data.update({
            "id_name": formatted_data.get('id', ""),  # Original file ID
            "dataset_name": formatted_data.get('dataset_name', ""),  # Dataset source
            "select_part": "compositional",
            "split_name": "vq",  # Split name for VQ training
            "motion_len": motion_len,  # Length of motion sequence
        })
        
        return formatted_data