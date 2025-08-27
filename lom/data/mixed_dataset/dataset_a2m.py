import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
# import spacy
import pandas as pd
import math
# from loguru import logger
# from utils_emage import rotation_conversions as rc
import torch
import smplx
from .data_tools import joints_list
import librosa
import textgrid as tg
from numpy.lib import stride_tricks
# import h5py
# import json
# import hashlib
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HAND,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
)
from tqdm import tqdm
from lom.utils.rotation_conversions import axis_angle_to_6d, axis_angle_to_matrix, axis_angle_to_6d_np



class Audio2MotionDataset(data.Dataset):

    def __init__(
        self,
        split,
        smpl_path,
        args,
        tiny=False,
        debug=False,
        audio_down = 320,
        use_cache=True,      # Whether to load data from cache
        save_cache=True,     # Whether to save processed data to cache
        cache_format="pkl",  # Format to use for caching: "h5", "npz", or "pkl"
        **kwargs,
    ):

        data_root = args.BEAT2.ROOT

        self.tiny = tiny
        self.debug = debug
        # self.unit_length = unit_length
        # We'll initialize SMPLX only when needed, not upfront
        self.smplx_2020 = None

        # Load each dataset based on its type from the configuration
        for config in args.datasets:
            dataset_type = config.get("name")
            if dataset_type == "BEAT2":
                self.testing_speakers = config.get('testing_speakers')
                code_path_audio = config.get("code_path_audio")
                config_beat2 = config


        self.args = args
        self.ori_length = self.args.test_length
        self.test_length = self.args.test_length
        self.audio_fps = args.audio_fps
        self.ori_joint_list = joints_list["beat_smplx_joints"]
        self.tar_joint_list = joints_list["beat_smplx_full"]
        self.smpl_path = smpl_path

        self.joints = 55
        self.joint_mask_upper = JOINT_MASK_UPPER
        self.joint_mask_lower = JOINT_MASK_LOWER
        self.joint_mask_hand = JOINT_MASK_HAND
        self.joint_mask_face = JOINT_MASK_FACE
        self.joint_mask_full = JOINT_MASK_FULL

        # Data path
        motion_dir = pjoin(data_root, "smplxflame_30")
        audio_dir = pjoin(data_root, code_path_audio)
        raw_audio_dir = pjoin(data_root, 'wave16k')

        # self.audio_down = 640  # 320
        self.audio_down = float(audio_down)  # 320

        self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = len(list(self.tar_joint_list.keys()))
        for joint_name in self.tar_joint_list:
            self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        # Make sure SMPLX is initialized
        self._initialize_smplx_if_needed()

        split_rule = pd.read_csv(pjoin(data_root,"train_test_split.csv"))
        self.selected_file = split_rule.loc[(split_rule['type'] == split) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.testing_speakers))]

        # Data id list
        self.id_list = []
        for index, file_name in self.selected_file.iterrows():
            self.id_list.append(file_name["id"])

        self.max_length = int(self.test_length)
        self.max_audio_pre_len = math.floor(self.test_length / args.pose_fps * self.audio_fps / self.audio_down)
        if self.max_audio_pre_len > self.test_length * self.audio_fps:
            self.max_audio_pre_len = self.test_length * self.audio_fps

        # Debug mode
        if tiny or debug:
            maxdata = 2
        else:
            maxdata = 1e10

        new_name_list = []
        length_list = []
        data_dict = {}

        # Set up cache parameters
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format
        self.split = split

        # Try to load from cache first
        cache_path = self._get_cache_path(config_beat2, data_root)
        if self.use_cache:
            cached_data = self._load_from_cache(cache_path, "BEAT2")
            if cached_data is not None:
                logger.info(f"Loaded dataset from cache: {cache_path}")
                self.length_arr = cached_data['length_arr']
                self.data_dict = cached_data['data_dict']
                self.name_list = cached_data['name_list']
                return  # Skip the rest of initialization if loaded from cache

        for name in tqdm(self.id_list):
            if len(new_name_list) > maxdata:
                break
            try:
                # Process each sample
                sample_data = self._process_sample(name, data_root, motion_dir, audio_dir, raw_audio_dir,self.audio_fps)
                data_dict[name] = sample_data
                new_name_list.append(name)
                length_list.append(len(sample_data['face']))
            except Exception as e:
                logger.warning(f"Error processing {name}: {str(e)}")
                continue

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        # Save to cache if specified
        if self.save_cache:
            cache_data = {
                'length_arr': length_list,
                'data_dict': data_dict,
                'name_list': new_name_list
            }
            self._save_to_cache(cache_path, cache_data, "BEAT2")


        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list


    def _initialize_smplx_if_needed(self):
        """
        Initialize the SMPLX model if it hasn't been initialized yet.
        This is done lazily to avoid unnecessary GPU memory usage when loading from cache.
        """
        if self.smplx_2020 is None:
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
        

    def _get_cache_path(self, config, data_root):
        """
        Generate a cache file path based on dataset configuration.
        
        Parameters:
        - config: Configuration dictionary for the dataset
        
        Returns:
        - Path to the cache file
        """
        # Get dataset type and root path from config
        dataset_type = config.get("name")
        
        # Create cache directory in the dataset's own directory
        os_cache_dir = os.path.join(data_root, 'cache')
        os.makedirs(os_cache_dir, exist_ok=True)
        
        # Generate a unique filename based on dataset configuration
        if dataset_type == "BEAT2":
            code_path = config.get("code_path")
            code_path_audio = config.get("code_path_audio")
            additional_data = config.get("additional_data")
            training_speakers = config.get("training_speakers")
            # pose_length = config.get("pose_length")
            
            # Format training speakers list in a more readable way
            if len(training_speakers) > 0:
                if all(isinstance(x, int) for x in training_speakers):
                    # Check if it's a continuous range
                    if list(range(min(training_speakers), max(training_speakers) + 1)) == sorted(training_speakers):
                        speakers_str = f"Speaker_{min(training_speakers)}_{max(training_speakers)}"
                    else:
                        # For non-continuous ranges, use abbreviated format
                        speakers_str = f"Speakers_{len(training_speakers)}"
                else:
                    speakers_str = "CustomSpeakers"
            else:
                speakers_str = "NoSpeakers"
            
            # Format boolean values more descriptively
            # vary_length_str = "VaryLength" if self.vary_length else "NoVaryLength"
            debug_str = "Debug" if self.debug else "NoDebug"
            
            config_str = f"{dataset_type}_{self.split}_lm_{self.test_length}_{code_path}_{code_path_audio}_{additional_data}_{speakers_str}_{debug_str}"
        
        elif dataset_type == "AMASS":
            code_path = config.get("code_path")
            code_path_audio = config.get("code_path_audio")
            # pose_length = config.get("pose_length")
            
            # Format boolean values more descriptively
            debug_str = "Debug" if self.debug else "NoDebug"
            
            config_str = f"{dataset_type}_{self.split}_lm_{self.test_length}_{code_path}_{code_path_audio}_{debug_str}"
        
        elif dataset_type == "librispeech":
            # Format boolean values more descriptively
            vary_length_str = "VaryLength" if self.vary_length else "NoVaryLength"
            debug_str = "Debug" if self.debug else "NoDebug"
            
            config_str = f"{dataset_type}_{self.split}_lm_{self.code_path_audio}_{vary_length_str}_{debug_str}"
        
        else:
            raise NotImplementedError(f"dataset_type {dataset_type} not implemented")
        
        # Set the file extension based on the cache format
        ext = ".pkl"  # We're using pickle format for simplicity and reliability
        
        cache_path = os.path.join(os_cache_dir, f"{config_str}{ext}")

        return cache_path 

    def _save_to_cache(self, cache_path, data_dict, dataset_name):
        """
        Save processed data to cache.
        
        Parameters:
        - cache_path: Path where to save the cache file
        - data_dict: Dictionary containing processed data
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
                if isinstance(item, dict):
                    numpy_item = {}
                    for attr_key, attr_value in item.items():
                        if isinstance(attr_value, torch.Tensor):
                            numpy_item[attr_key] = attr_value.cpu().numpy()
                        else:
                            numpy_item[attr_key] = attr_value
                    numpy_data_dict[key] = numpy_item
                else:
                    # Handle non-dict items like length_arr
                    if isinstance(item, torch.Tensor):
                        numpy_data_dict[key] = item.cpu().numpy()
                    else:
                        numpy_data_dict[key] = item
            
            # Simple pickle save
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'length_arr': data_dict['length_arr'],
                    'data_dict': data_dict['data_dict'],
                    'name_list': data_dict['name_list']
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            print(f"Successfully saved {dataset_name} cache to {cache_path}")
            
        except Exception as e:
            print(f"Error saving {dataset_name} cache: {str(e)}")
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
            return None
        
        if not os.path.exists(cache_path):
            print(f"Cache file not found: {cache_path}")
            return None
        
        try:
            print(f"Loading {dataset_name} dataset from cache: {cache_path}")
            
            # Simple pickle load
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                # data_dict = cached_data['data_dict']
                # metadata = cached_data['metadata']
                # length_arr = cached_data['length_arr']

            print(f"Successfully loaded {dataset_name} cache from {cache_path}")
            return cached_data
            
        except Exception as e:
            print(f"Error loading {dataset_name} cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return None


    def _process_sample(self, name, data_root, motion_dir, audio_dir, raw_audio_dir, audio_fps):
        """
        Process a single sample from the dataset.
        
        Parameters:
        - name: Name of the sample
        - data_root: Root directory of the dataset
        - motion_dir: Directory containing motion data
        - audio_dir: Directory containing audio data
        - audio_fps: Audio frame rate
        
        Returns:
        - Processed sample data
        """
        pose_file = pjoin(motion_dir, f'{name}.npz')

        # Process raw audio
        raw_audio_eval, sr = librosa.load(pjoin(raw_audio_dir, f'{name}.wav'))
        raw_audio_eval = librosa.resample(raw_audio_eval, orig_sr=sr, target_sr=audio_fps)

        # # Process amplitude envelope
        # frame_length = 1024
        # shape = (raw_audio_eval.shape[-1] - frame_length + 1, frame_length)
        # strides = (raw_audio_eval.strides[-1], raw_audio_eval.strides[-1])
        # rolling_view = stride_tricks.as_strided(raw_audio_eval, shape=shape, strides=strides)
        # amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
        # amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length - 1), mode='constant',
        #                             constant_values=amplitude_envelope[-1])

        # amplitude_envelope_norm = (amplitude_envelope - amplitude_envelope.min()) / (amplitude_envelope.max() - amplitude_envelope.min())
        # amplitude_envelope_discrete = np.round(amplitude_envelope_norm * (512 - 1)).astype(int)

        # # Subsample factor
        # factor = int(self.audio_down)
        # amplitude_envelope_downsampled = amplitude_envelope_discrete[::factor]
        # amplitude_envelope_downsampled = torch.from_numpy(amplitude_envelope_downsampled)

        # # Process onset information
        # audio_onset_f = librosa.onset.onset_detect(y=raw_audio_eval, sr=self.args.audio_fps, units='frames')
        # onset_array = np.zeros(len(raw_audio_eval), dtype=float)
        # onset_array[audio_onset_f] = 1.0
        # onset_array_downsampled = onset_array[::factor]
        # onset_array_downsampled = torch.from_numpy(onset_array_downsampled)

        # Load motion data
        pose_data = np.load(pose_file, allow_pickle=True)
        n, c = pose_data["poses"].shape[0], pose_data["poses"].shape[1]
        assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
        stride = int(30/self.args.pose_fps)
        pose_each_file = pose_data["poses"][::stride]
        trans_each_file = pose_data["trans"][::stride]
        shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0)

        # Process SMPLX data
        # m_data = np.load(pose_file, allow_pickle=True)
        # Calculate foot contacts
        foot_contacts_path = pjoin(data_root, 'foot_contacts', name + '.npy')
        if os.path.exists(foot_contacts_path):
            contacts = np.load(foot_contacts_path)
        else:
            contacts = self.comput_foot_contacts(pose_data)
            os.makedirs(pjoin(data_root, 'foot_contacts'), exist_ok=True)
            np.save(foot_contacts_path, contacts)
        # pose_processed = np.concatenate([pose_processed, contacts], axis=1)
        
        # Apply joint mask and add contacts
        pose_each_file = pose_each_file * self.joint_mask
        pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
        pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)

        # Process facial expressions
        if hasattr(self.args, 'facial_rep') and self.args.facial_rep is not None:
            facial_each_file = pose_data["expressions"][::stride]
            if hasattr(self.args, 'facial_norm') and self.args.facial_norm:
                facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
        else:
            facial_each_file = pose_data["expressions"][::stride]


        tar_pose = pose_each_file[:, :165]
        tar_contact = pose_each_file[:, 165:169]
        tar_exps = facial_each_file
        tar_trans = trans_each_file

        # Process different body parts
        # Jaw pose (face)
        tar_pose_jaw = tar_pose[:, 66:69]
        tar_pose_jaw_6d = axis_angle_to_6d_np(tar_pose_jaw).reshape(n, 6)
        # Concatenate jaw pose and expressions for face
        tar_pose_face = np.concatenate([tar_pose_jaw_6d, tar_exps], axis=1)

        # Extract and convert hand pose data
        tar_pose_hands = tar_pose[:, 25 * 3:55 * 3].reshape(n, 30, 3)
        tar_pose_hands_6d = axis_angle_to_6d_np(tar_pose_hands).reshape(n, 30 * 6)

        # Extract and convert upper body pose data
        tar_pose_upper = tar_pose[:, self.joint_mask_upper.astype(bool)].reshape(n, 13, 3)
        tar_pose_upper_6d = axis_angle_to_6d_np(tar_pose_upper).reshape(n, 13 * 6)

        # Extract and convert lower body pose data
        tar_pose_leg = tar_pose[:, self.joint_mask_lower.astype(bool)].reshape(n, 9, 3)
        tar_pose_leg_6d = axis_angle_to_6d_np(tar_pose_leg).reshape(n, 9 * 6)
        
        # Convert other data to tensors
        tar_pose_lower = np.concatenate([tar_pose_leg_6d, tar_trans, tar_contact], axis=1)
        tar_pose_6d = axis_angle_to_6d_np(tar_pose.reshape(n, 55, 3)).reshape(n, 55 * 6)

        # Load audio data
        audio = np.load(pjoin(audio_dir, name + ".npy"))
        audio = torch.from_numpy(audio)

        # Process text timestamps
        word_file = pjoin(data_root, 'textgrid', name + ".TextGrid")
        tgrid = tg.TextGrid.fromFile(word_file)
        audio_length = audio.shape[0]
        text_with_timestamps = []
        
        for word_index in range(audio_length):
            found_flag = False
            current_time = word_index / (audio_fps / self.audio_down)
            for j, word in enumerate(tgrid[0]):
                word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                if word_s <= current_time and current_time <= word_e:
                    if word_n == '':
                        text_with_timestamps.append("None")
                    else:
                        text_with_timestamps.append(word_n)
                    found_flag = True
                    break
            if not found_flag:
                text_with_timestamps.append("None")

        # Return processed data
        return {
            'face': tar_pose_face,
            'hand': tar_pose_hands_6d,
            'upper': tar_pose_upper_6d,
            'lower': tar_pose_lower,
            'tar_pose': tar_pose_6d,
            'tar_beta': shape_each_file,
            'tar_trans': tar_trans,
            'tar_exps': tar_exps,
            'text_timestamp': text_with_timestamps,
            'audio': audio,
            'raw_audio': raw_audio_eval,
            # 'onset': onset_array_downsampled,
            # 'amplitude_envelope': amplitude_envelope_downsampled,
        }


    def comput_foot_contacts(self, m_data):
        """
        Compute foot contacts from motion data.
        This method requires SMPLX, so we ensure it's initialized.
        """
        # Make sure SMPLX is initialized
        self._initialize_smplx_if_needed()
        
        betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
        n, c = poses.shape[0], poses.shape[1]
        
        # determine the dimension of betas
        beta_dim = betas.shape[-1]  # get the last dimension of betas

        if beta_dim == 16:
            padded_betas = np.zeros(300)
            padded_betas[:16] = betas
            exps = torch.zeros([n, 100], dtype=torch.float32).cuda()  # AMASS dataset
        else:
            padded_betas = betas
            exps = torch.from_numpy(m_data["expressions"]).cuda().float()  # BEAT2 dataset

        betas = padded_betas.reshape(1, 300)  # can be 16 or 300
        betas = np.tile(betas, (n, 1))

        betas = torch.from_numpy(betas).cuda().float()
        poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
        trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
        max_length = 128
        s, r = n // max_length, n % max_length
        all_tensor = []
        
        for i in range(s):
            with torch.no_grad():
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
                )['joints'][:, (7, 8, 10, 11), :].reshape(max_length, 4, 3).cpu()
            all_tensor.append(joints)
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
        joints = torch.cat(all_tensor, axis=0)  # all, 4, 3
        feetv = torch.zeros(joints.shape[1], joints.shape[0])
        joints = joints.permute(1, 0, 2)
        feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).numpy().astype(float)
        contacts = contacts.transpose(1, 0)

        return contacts


    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list)


    def _ensure_tensor(self, data):
        """
        Ensures the data is a PyTorch tensor.
        If it's a numpy array, converts it to a tensor.
        Otherwise, returns the original data.
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
            return torch.tensor(data, dtype=torch.float)
        elif not isinstance(data, torch.Tensor):
            # If it's neither a tensor nor a numpy array nor a numeric list
            # but it's supposed to be a tensor, log a warning
            if data is not None and not isinstance(data, (str, list)):
                print(f"Warning: Expected tensor but got {type(data)}")
        
        return data



    def __getitem__(self, item):
        # Get text data

        data = self.data_dict[self.name_list[item]]
        
        # Extract all data fields
        face = self._ensure_tensor(data['face'])
        hand = self._ensure_tensor(data['hand'])
        lower = self._ensure_tensor(data['lower'])
        upper = self._ensure_tensor(data['upper'])
        tar_pose = self._ensure_tensor(data['tar_pose'])
        tar_beta = self._ensure_tensor(data['tar_beta'])
        tar_trans = self._ensure_tensor(data['tar_trans'])
        tar_exps = self._ensure_tensor(data['tar_exps'])
        
        # # Audio data
        raw_audio = data['raw_audio']  # Keep raw audio as numpy for compatibility
        audio_token = self._ensure_tensor(data['audio'])
        
        # Calculate lengths
        m_tokens_len = torch.tensor(face.shape[0])
        a_tokens_len = audio_token.shape[0]
        
        # Text data (keep as is)
        text_timestamp = data['text_timestamp']

        return {
            "face": face, 
            "hand": hand, 
            "lower": lower, 
            "upper": upper, 
            "tar_pose": tar_pose,
            "tar_beta": tar_beta, 
            "tar_trans": tar_trans, 
            "tar_exps": tar_exps, 
            "audio_token": audio_token,
            "raw_audio": raw_audio, 
            "m_tokens_len": m_tokens_len, 
            "a_tokens_len": a_tokens_len, 
            "split_name": 'test', 
            "text_timestamp": text_timestamp
        }
