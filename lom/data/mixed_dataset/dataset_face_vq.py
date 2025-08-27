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
    JOINT_MASK_FACE,
    JOINT_MASK_FULL,
)
from lom.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, rotation_6d_to_axis_angle, axis_angle_to_6d_np

class FaceDatasetVQ(data.Dataset):
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
        self.joint_mask_face = JOINT_MASK_FACE


        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []
        # Load each dataset based on its type from the configuration
        for config in dataset_configs:
            # dataset_name = config.get("type")
            dataset_name = config.get("name")
            if dataset_name == "BEAT2":
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

    # def _initialize_smplx_if_needed(self):
    #     """
    #     Initialize the SMPLX model if it hasn't been initialized yet.
    #     This is done lazily to avoid unnecessary GPU memory usage when loading from cache.
    #     """
    #     if self.smplx_2020 is None and (self.motion_representation == "rotation"):
    #         print("Initializing SMPLX model for data processing...")
    #         self.smplx_2020 = smplx.create(self.smpl_path,
    #             model_type='smplx',
    #             gender='NEUTRAL_2020',
    #             use_face_contour=False,
    #             num_betas=300,
    #             num_expression_coeffs=100,
    #             ext='npz',
    #             use_pca=False,
    #             ).cuda().eval()

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
        
        # # We need to process data from scratch, so initialize SMPLX if needed
        # self._initialize_smplx_if_needed()
        
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
            betas = pose_data["betas"]
            # Repeat betas for each frame
            betas = np.repeat(pose_data["betas"].reshape(1, 300), poses.shape[0], axis=0)

            expressions = pose_data["expressions"]

            # Apply joint mask to filter relevant joints
            pose_processed = poses * JOINT_MASK_FULL
            pose_processed = pose_processed[:, JOINT_MASK_FULL.astype(bool)]
            
            # Extract different components from processed pose
            tar_pose = pose_processed[ :, :165]  # Body pose (55 joints * 3)
            tar_exps = expressions  # Facial expressions

            # Extract and convert jaw pose data
            tar_pose_jaw = tar_pose[:, 66:69]  # Jaw pose (3D rotation)
            tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)  # Convert to 6D representation
            
            # Concatenate jaw pose and expressions for face data
            tar_pose_face = np.concatenate([tar_pose_jaw_6d, tar_exps], axis=1)

            # Calculate time segments
            round_seconds_skeleton = tar_pose_face.shape[0] // pose_fps_beat2
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
                sample_face = tar_pose_face[start_idx:fin_idx]
                sample_shape = betas[start_idx:fin_idx]

                # Create unique name for this segment
                new_name = 'beat2_' + '%s_%d' % (f_name,i)
                
                # Store processed data
                self.data_dict_beat2[new_name] = {
                    'face': sample_face,
                    'shape': sample_shape,
                    'id': f_name,
                    'dataset_name': 'beat2',
                }
                self.metadata_beat2.append(new_name)
                
            # Break early for debugging if max data reached
            if index >= self.maxdata:
                break
        
        # Save processed data to cache
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2")

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
        motion_len = formatted_data['face'].shape[0] if 'face' in formatted_data else 0
        
        # Add additional information needed for training
        formatted_data.update({
            "id_name": formatted_data.get('id', ""),  # Original file ID
            "select_part": "face",
            "dataset_name": formatted_data.get('dataset_name', ""),  # Dataset source
            "split_name": "vq",  # Split name for VQ training
            "motion_len": motion_len,  # Length of motion sequence
        })
        
        return formatted_data