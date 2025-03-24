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
from .utils.split_transcript import split_and_merge_sentences
import librosa
from numpy.lib import stride_tricks

class MixedDatasetCB(data.Dataset):
    def __init__(
        self,
        dataset_configs,
        split,
        args,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        task_path=None,
        audio_down=320,
        use_cache=True,  # Whether to load data from cache when available
        save_cache=True,  # Whether to save processed data to cache
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
        - std_text: Whether to standardize text.
        """
        # Set max data size depending on debug mode
        self.debug = debug
        if tiny or debug:
            self.maxdata = 10
        else:
            self.maxdata = 1e10

        self.stage = stage
        self.task_path = task_path
        # self.unit_length = unit_length
        self.args = args
        self.vary_length = args.vary_length
        # self.ori_length = self.args.pose_length
        self.split = split
        self.audio_down = float(audio_down)  # 320
        self.test_length = args.test_length
        self.audio_fps = args.audio_fps
        # Dictionary to store data and metadata
        self.data_dict = {}
        self.metadata = []

        # Add cache-related attributes
        self.use_cache = use_cache
        self.save_cache = save_cache
        self.cache_format = cache_format

        # Load each dataset based on its type from the configuration
        for config in dataset_configs:
            dataset_name = config.get("name")
            if dataset_name == "LibriSpeech":
                self._load_librispeech(config)
                self.data_dict.update(self.data_dict_librispeech)
                self.metadata.extend(self.metadata_librispeech)
                print(f"Loaded LibriSpeech dataset with {len(self.metadata_librispeech)} samples")
            elif dataset_name == "AMASS":
                self._load_amass(config)
                self.data_dict.update(self.data_dict_amass)
                self.metadata.extend(self.metadata_amass)
                print(f"Loaded AMASS dataset with {len(self.metadata_amass)} samples")
            elif dataset_name == "BEAT2":
                self._load_beat2(config)
                self.data_dict.update(self.data_dict_beat2)
                self.metadata.extend(self.metadata_beat2)
                print(f"Loaded BEAT2 dataset with {len(self.metadata_beat2)} samples")
            else:
                raise NotImplementedError(f"Unknown dataset name {dataset_name}")

        # Load the language model for text processing
        # self.nlp = spacy.load('en_core_web_sm')
        # self.std_text = std_text

    def _load_librispeech(self, config):
        """
        Load the LibriSpeech dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for LibriSpeech.
        """
        data_root = self.args["LIBRISPEECH"].ROOT
        cache_path = self._get_cache_path(config, data_root)

                # Try to load from cache first
        data_dict, metadata = self._load_from_cache(cache_path, "LibriSpeech")
        if data_dict is not None:
            self.data_dict_librispeech = data_dict
            self.metadata_librispeech = metadata
            return
        
        code_path_audio = config.get("code_path_audio")
        token_root_librispeech = pjoin(data_root, code_path_audio)
        instructions_file = config.get("instructions_file")
        used_trainset = config.get("used_trainset")

        ##################  LibriSpeech  ##################
        # Determine the instructions path based on the stage
        if self.task_path:
            instructions_path = self.task_path
        elif self.stage == 'lm_pretrain':
            instructions_path = pjoin(data_root, instructions_file)
        elif self.stage in ['lm_instruct', 'lm_causal_instruct']:
            instructions_path = pjoin(data_root, instructions_file)
        else:
            raise NotImplementedError(f"stage {self.stage} not implemented")

        # Load instructions and tasks for LibriSpeech
        self.instructions_librispeech = json.load(open(instructions_path, 'r'))
        self.tasks_librispeech = []
        for task in self.instructions_librispeech.keys():
            for subtask in self.instructions_librispeech[task].keys():
                self.tasks_librispeech.append(self.instructions_librispeech[task][subtask])

        # Pre-load all text and token data
        self.text_dict_librispeech = {}
        self.token_dict_librispeech = {}
        self.metadata_librispeech_original = []

        # Load data for each specified trainset
        for trainset in used_trainset:
            # Load text and token data from pickle files
            with open(pjoin(token_root_librispeech, trainset + '-texts.pkl'), 'rb') as f:
                self.text_dict_librispeech[trainset] = pickle.load(f)

            with open(pjoin(token_root_librispeech, trainset + '.pkl'), 'rb') as f:
                self.token_dict_librispeech[trainset] = pickle.load(f)

            # Collect metadata for each sample in the dataset
            for speaker_id in self.token_dict_librispeech[trainset].keys():
                for chapter_id in self.token_dict_librispeech[trainset][speaker_id].keys():
                    for utterance_id in self.token_dict_librispeech[trainset][speaker_id][chapter_id].keys():
                        for part_id in self.token_dict_librispeech[trainset][speaker_id][chapter_id][utterance_id].keys():
                            self.metadata_librispeech_original.append((trainset, speaker_id, chapter_id, utterance_id, part_id))

            # Prepare metadata and data dictionary for LibriSpeech
            enumerator_librispeech = enumerate(self.metadata_librispeech_original)
            self.metadata_librispeech = []
            self.data_dict_librispeech = {}
            for i, name in enumerator_librispeech:
                if len(self.metadata_librispeech) > self.maxdata:
                    break
                try:
                    # Retrieve metadata for the current item
                    trainset, speaker_id, chapter_id, utterance_id, part_id = name
                    # Get the corresponding token and text data
                    a_token = self.token_dict_librispeech[trainset][speaker_id][chapter_id][utterance_id][part_id]
                    text = self.text_dict_librispeech[trainset][speaker_id][chapter_id][utterance_id][part_id]
                    # Get the length of the token data
                    a_tokens_len = a_token.shape[0]

                    # Create a unique save name for the data entry
                    for tasks in self.tasks_librispeech:
                        save_name = 'librispeech' + '_' + trainset + '_' + speaker_id + '_' + chapter_id + '_' + utterance_id + '_' + part_id + '_' + tasks['class']
                        # Store the data entry
                        self.data_dict_librispeech[save_name] = {
                            'face_token': None,
                            'hand_token': None,
                            'lower_token': None,
                            'upper_token': None,
                            'text': text,
                            'audio': a_token,
                            'tasks': tasks,
                            'emotion_label': None,
                        }
                        self.metadata_librispeech.append(save_name)

                except:
                    pass

        # Save to cache after processing
        self._save_to_cache(cache_path, self.data_dict_librispeech, self.metadata_librispeech, "LibriSpeech")

    def _load_beat2(self, config):
        """
        Load the BEAT2 dataset based on the configuration.
        """
        # Get cache path for this specific dataset
        # data_root_beat2 = config.get("data_root")
        data_root = self.args["BEAT2"].ROOT
        cache_path = self._get_cache_path(config, data_root)
        
        # Try to load from cache first
        data_dict, metadata = self._load_from_cache(cache_path, "BEAT2")
        if data_dict is not None:
            self.data_dict_beat2 = data_dict
            self.metadata_beat2 = metadata
            return
            
        print(f"Processing BEAT2 dataset...")
        
        code_path = config.get("code_path")
        instructions_file = config.get("instructions_file")
        training_speakers = config.get("training_speakers")
        code_path_audio = config.get("code_path_audio")
        # vary_length = config.get("vary_length")
        additional_data = config.get("additional_data")
        # pose_length = config.get("pose_length")
        # test_length = config.get("test_length")
        # audio_fps = config.get("audio_fps")
        stride = config.get("stride")
        # Determine the instructions path based on the stage
        if self.task_path:
            instructions_path = self.task_path
        elif self.stage in ['lm_pretrain']:
            instructions_path = pjoin(data_root, instructions_file)
        elif self.stage in ['lm_instruct', "lm_causal_instruct"]:
            instructions_path = pjoin(data_root, instructions_file)
        else:
            raise NotImplementedError(f"stage {self.stage} not implemented")

        # Load instructions and tasks for BEAT2
        self.instructions_beat2 = json.load(open(instructions_path, 'r'))
        self.tasks_beat2 = []
        for task in self.instructions_beat2.keys():
            for subtask in self.instructions_beat2[task].keys():
                self.tasks_beat2.append(self.instructions_beat2[task][subtask])

        ##################  BEAT2  ##################
        # Define lengths for data processing
        audio_short_length = int(self.test_length / self.args.pose_fps * self.audio_fps / self.audio_down)

        # Define data paths
        motion_dir_beat2 = pjoin(data_root, code_path)
        text_dir_beat2 = pjoin(data_root, 'texts')
        audio_dir_beat2 = pjoin(data_root, code_path_audio)
        raw_audio_dir = pjoin(data_root, 'wave16k')

        split_rule = pd.read_csv(pjoin(data_root,"train_test_split.csv"))

        # Select files based on the split and training speakers
        self.selected_file = split_rule.loc[(split_rule['type'] == self.split) & (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))]
        if additional_data and self.split == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])

        # Calculate maximum lengths for data
        self.max_length = int(self.test_length)
        self.max_audio_pre_len = math.floor(self.test_length / self.args.pose_fps * self.audio_fps / self.audio_down)
        if self.max_audio_pre_len > self.test_length * self.audio_fps:
            self.max_audio_pre_len = self.test_length * self.audio_fps

        # Create a list of data IDs
        self.id_list = []
        for index, file_name in self.selected_file.iterrows():
            self.id_list.append(file_name["id"])

        if 'data_rate' in config:
            data_rate = config.get("data_rate")
            maxdata_beat2 = round( len(self.id_list) * data_rate)
        else:
            data_rate = 1
            maxdata_beat2 = self.maxdata

        # Iterate through the files to load data
        enumerator = enumerate(track(self.id_list, f"Loading Beat2 {self.split}"))

        self.metadata_beat2 = []
        self.data_dict_beat2 = {}

        squence_num = 0
        # Load motion data, audio, and associated metadata
        for i, name in enumerator:
            # if len(self.metadata_beat2) > maxdata_beat2:
            #     break
            if squence_num > maxdata_beat2:
                break
            try:

                # Load motion tokens
                m_token_face = np.load(pjoin(motion_dir_beat2, 'face', f'{name}.npy'))
                m_token_hands = np.load(pjoin(motion_dir_beat2, 'hands', f'{name}.npy'))
                m_token_lower = np.load(pjoin(motion_dir_beat2, 'lower', f'{name}.npy'))
                m_token_upper = np.load(pjoin(motion_dir_beat2, 'upper', f'{name}.npy'))
                audio = np.load(pjoin(audio_dir_beat2, name + ".npy"))

                emotion_data = pd.read_csv(pjoin(data_root, 'emotion_label', name + ".csv"))
                emotion_label = emotion_data.columns[0].split('_')[-1]
                emotion_start_time = float(emotion_data.columns[1].split('_')[-1])
                emotion_stop_time = float(emotion_data.columns[2].split('_')[-1])

                # emotion_label = pd.read_csv(pjoin(data_root_beat2, 'emotion_label', name + ".csv")).columns[0].split('_')[-1]
                # Load corresponding text annotations from TextGrid
                word_file = pjoin(data_root, 'textgrid', name + ".TextGrid")
                tgrid = tg.TextGrid.fromFile(word_file)
                motion_length = m_token_face.shape[1]


                if self.vary_length == True:
                    # Call the function using your tgrid
                    ## Since the maximum number of text tokens in instruction tuning template is 52. So setting the audio less than 400 is safe.
                    ## But if the task ask for taking both audio and transcript as inpput, the combining length will longer than 512.
                    paragraphs = split_and_merge_sentences(tgrid[0].intervals, max_duration=4.0)   ### Consider the combined string, I set 4 for test
                    clip_index = 0
                    # Split data into smaller chunks for training
                    for text, start_time, end_time in paragraphs:
                        start_idx = int(start_time * self.args.pose_fps)
                        fin_idx = int(end_time * self.args.pose_fps)
                        audio_start = int(start_time * self.audio_fps / self.audio_down)
                        audio_end = int(end_time * self.audio_fps / self.audio_down)
                        sample_face = m_token_face[0, start_idx:fin_idx]
                        sample_hand = m_token_hands[0, start_idx:fin_idx]
                        sample_lower = m_token_lower[0, start_idx:fin_idx]
                        sample_upper = m_token_upper[0, start_idx:fin_idx]
                        sample_audio = audio[audio_start:audio_end]

                        # Save data entries
                        for tasks in self.tasks_beat2:
                            new_name = 'beat2_' + name + '_' + str(clip_index) + '_' + tasks['class']
                            self.data_dict_beat2[new_name] = {
                                'face_token': sample_face,
                                'hand_token': sample_hand,
                                'lower_token': sample_lower,
                                'upper_token': sample_upper,
                                'text': text,
                                'audio': sample_audio,
                                'tasks': tasks,
                                'emotion_label':emotion_label
                            }
                            self.metadata_beat2.append(new_name)
                        clip_index += 1
                else:
                    # Compute the start and end times for clipping
                    round_seconds_skeleton = motion_length // self.args.pose_fps
                    clean_first_seconds = 0
                    clean_final_seconds = 0
                    clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds
                    clip_s_f_audio, clip_e_f_audio = int(self.audio_fps * clip_s_t / self.audio_down), int(self.audio_fps * clip_e_t / self.audio_down)
                    clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps

                    num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - self.test_length) / stride) + 1

                    # Split data into smaller chunks for training
                    for clip_index in range(num_subdivision):
                        start_idx = clip_s_f_pose + clip_index * stride
                        start_time = start_idx / self.args.pose_fps * 1000.

                        audio_start = clip_s_f_audio + math.floor(clip_index * stride * self.audio_fps / self.audio_down / self.args.pose_fps)


                        fin_idx = start_idx + self.test_length
                        fin_time = fin_idx / self.args.pose_fps * 1000.
                        audio_end = audio_start + audio_short_length

                        if start_time >= 1000. * emotion_start_time and fin_time <= 1000. * emotion_stop_time:
                            emotion_label_temp = emotion_label
                        else:
                            emotion_label_temp = 'unknown'


                        sample_face = m_token_face[0, start_idx:fin_idx]
                        sample_hand = m_token_hands[0, start_idx:fin_idx]
                        sample_lower = m_token_lower[0, start_idx:fin_idx]
                        sample_upper = m_token_upper[0, start_idx:fin_idx]
                        sample_audio = audio[audio_start:audio_end]

                        # Extract text within the time range
                        text_candidate = [interval.mark
                                          for interval in tgrid[0].intervals
                                          if float(interval.minTime) * 1000 >= start_time and float(interval.maxTime) * 1000 <= fin_time and interval.mark is not None]

                        text = " ".join(text_candidate)

                        # Save data entries
                        for tasks in self.tasks_beat2:

                            if tasks['class'] == 'fhul2e' and emotion_label_temp == 'unknown':
                                continue

                            new_name = 'beat2_' + name + '_' + str(clip_index) + '_' + tasks['class']
                            self.data_dict_beat2[new_name] = {
                                'face_token': sample_face,
                                'hand_token': sample_hand,
                                'lower_token': sample_lower,
                                'upper_token': sample_upper,
                                'text': text,
                                'audio': sample_audio,
                                'tasks': tasks,
                                'emotion_label': emotion_label_temp
                            }
                            self.metadata_beat2.append(new_name)

                squence_num += 1

            except:
                print(f"Error loading {name}")
                pass

        # Save to cache after processing
        self._save_to_cache(cache_path, self.data_dict_beat2, self.metadata_beat2, "BEAT2")

    def _load_amass(self, config):
        """
        Load the AMASS dataset based on the configuration.

        Parameters:
        - config: Configuration dictionary for AMASS.
        """
        data_root = self.args["AMASS"].ROOT

        cache_path = self._get_cache_path(config, data_root)
        
        # Try to load from cache first
        data_dict, metadata = self._load_from_cache(cache_path, "AMASS")
        if data_dict is not None:
            self.data_dict_amass = data_dict
            self.metadata_amass = metadata
            return

        code_path = config.get("code_path")
        instructions_file = config.get("instructions_file")
        amass_max_length = config.get("max_length")
        amass_min_length = config.get("min_length")

        # Determine the instructions path based on the stage
        if self.task_path:
            instructions_path = self.task_path
        elif self.stage == 'lm_pretrain':
            instructions_path = pjoin(data_root, instructions_file)
        elif self.stage in ['lm_instruct', "lm_causal_instruct"]:
            instructions_path = pjoin(data_root, instructions_file)
        else:
            raise NotImplementedError(f"stage {self.stage} not implemented")

        # Load instructions and tasks for AMASS
        self.instructions_amass = json.load(open(instructions_path, 'r'))
        self.tasks_amass = []
        for task in self.instructions_amass.keys():
            for subtask in self.instructions_amass[task].keys():
                self.tasks_amass.append(self.instructions_amass[task][subtask])

        ##################  AMASS  ##################
        # Define lengths for data processing

        # Define data paths
        motion_dir_amass = pjoin(data_root, code_path)
        text_dir_amass = pjoin(data_root, 'texts')
        # audio_dir_amass = pjoin(data_root_amass, 'audios_token')

        split_file = pjoin(data_root, self.split + '.txt')
        # Data id list
        id_list = []
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                id_list.append(line.strip())

        # Iterate through the files to load data
        enumerator = enumerate(track(id_list, f"Loading AMASS {self.split}"))
        self.metadata_amass = []
        self.data_dict_amass = {}

        # Fast loading
        for i, name in enumerator:
            if len(self.metadata_amass) > self.maxdata:
                break
            try:
                # Load motion tokens
                hand_token_list = np.load(pjoin(motion_dir_amass, 'hands', f'{name}.npy'))
                upper_token_list = np.load(pjoin(motion_dir_amass, 'upper', f'{name}.npy'))
                lower_token_list = np.load(pjoin(motion_dir_amass, 'lower', f'{name}.npy'))
                face_token_list = np.zeros_like(lower_token_list)

                if lower_token_list.shape[1] > amass_max_length or lower_token_list.shape[1] < amass_min_length:
                    continue

                # Read text
                with cs.open(pjoin(text_dir_amass, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:

                                hand_token_list_new = [
                                    tokens[int(f_tag * self.args.pose_fps):int(to_tag * self.args.pose_fps)]
                                    for tokens in hand_token_list
                                    if int(f_tag * self.args.pose_fps) < int(to_tag * self.args.pose_fps)
                                ]
                                upper_token_list_new = [
                                    tokens[int(f_tag * self.args.pose_fps):int(to_tag * self.args.pose_fps)]
                                    for tokens in upper_token_list
                                    if int(f_tag * self.args.pose_fps) < int(to_tag * self.args.pose_fps)
                                ]
                                lower_token_list_new = [
                                    tokens[int(f_tag * self.args.pose_fps):int(to_tag * self.args.pose_fps)]
                                    for tokens in lower_token_list
                                    if int(f_tag * self.args.pose_fps) < int(to_tag * self.args.pose_fps)
                                ]
                                face_token_list_new = [
                                    tokens[int(f_tag * self.args.pose_fps):int(to_tag * self.args.pose_fps)]
                                    for tokens in face_token_list
                                    if int(f_tag * self.args.pose_fps) < int(to_tag * self.args.pose_fps)
                                ]

                                if len(hand_token_list_new) == 0:
                                    continue

                                for tasks in self.tasks_amass:
                                    new_name = 'amass_' + '%s_%f_%f' % (name, f_tag, to_tag) + tasks['class']
                                    self.data_dict_amass[new_name] = {
                                        'face_token': face_token_list_new,
                                        'hand_token': hand_token_list_new,
                                        'lower_token': lower_token_list_new,
                                        'upper_token': upper_token_list_new,
                                        'text': [text_dict],
                                        'audio': None,
                                        'tasks': tasks
                                    }
                                    self.metadata_amass.append(new_name)

                        except:
                            pass

                if flag:
                    for tasks in self.tasks_amass:
                        save_name = 'amass_' + name + '_' + tasks['class']
                        self.data_dict_amass[save_name] = {
                            'face_token': face_token_list,
                            'hand_token': hand_token_list,
                            'lower_token': lower_token_list,
                            'upper_token': upper_token_list,
                            'text': text_data,
                            'audio': None,
                            'tasks': tasks
                        }
                        self.metadata_amass.append(save_name)
            except:
                pass


        # Save to cache after processing
        self._save_to_cache(cache_path, self.data_dict_amass, self.metadata_amass, "AMASS")

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
            vary_length_str = "VaryLength" if self.vary_length else "NoVaryLength"
            debug_str = "Debug" if self.debug else "NoDebug"
            
            config_str = f"{dataset_type}_{self.split}_{self.stage}_{self.test_length}_{code_path}_{code_path_audio}_{additional_data}_{speakers_str}_{vary_length_str}_{debug_str}"
        
        elif dataset_type == "AMASS":
            code_path = config.get("code_path")
            code_path_audio = config.get("code_path_audio")
            # pose_length = config.get("pose_length")
            # Format boolean values more descriptively
            debug_str = "Debug" if self.debug else "NoDebug"
            config_str = f"{dataset_type}_{self.split}_{self.stage}_{self.test_length}_{code_path}_{code_path_audio}_{debug_str}"
        
        elif dataset_type == "LibriSpeech":
            # Format boolean values more descriptively
            used_trainset = config.get("used_trainset")
            trainset_str = f"{used_trainset[0]}"
            for trainset in used_trainset[1:]:
                trainset_str += f"_{trainset}"

            vary_length_str = "VaryLength" if self.vary_length else "NoVaryLength"
            debug_str = "Debug" if self.debug else "NoDebug"
            
            config_str = f"{dataset_type}_{self.split}_{self.stage}_{trainset_str}_{vary_length_str}_{debug_str}"
        
        else:
            raise NotImplementedError(f"dataset_type {dataset_type} not implemented")
        
        # Set the file extension based on the cache format
        ext = ".pkl"  # We're using pickle format for simplicity and reliability
        cache_path = os.path.join(os_cache_dir, f"{config_str}{ext}")

        return cache_path 

    def _save_to_cache(self, cache_path, data_dict, metadata, dataset_name):
        """
        Save processed data to cache.
        
        Parameters:
        - cache_path: Path where to save the cache file
        - data_dict: Dictionary containing processed data
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
                        numpy_item[attr_key] = attr_value.cpu().numpy()
                    else:
                        numpy_item[attr_key] = attr_value
                numpy_data_dict[key] = numpy_item
            
            # Simple pickle save
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'data_dict': numpy_data_dict,
                    'metadata': metadata
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
            return None, None
        
        if not os.path.exists(cache_path):
            print(f"Cache file not found: {cache_path}")
            return None, None
        
        try:
            print(f"Loading {dataset_name} dataset from cache: {cache_path}")
            
            # Simple pickle load
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                data_dict = cached_data['data_dict']
                metadata = cached_data['metadata']
            
            print(f"Successfully loaded {dataset_name} cache from {cache_path}")
            return data_dict, metadata
            
        except Exception as e:
            print(f"Error loading {dataset_name} cache: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.metadata)

    def __getitem__(self, item):
        """
        Retrieves a sample from the dataset based on the given index.

        Parameters:
        - item: Index of the sample to retrieve.

        Returns:
        - A dictionary containing motion tokens, audio tokens, tasks, and other information.
        """

        data_name = self.metadata[item]
        data = self.data_dict[data_name]
        face_token, hand_token, lower_token, upper_token, audio_token = data['face_token'], data['hand_token'], data['lower_token'], data['upper_token'], data['audio']


        if data_name.split('_')[0] == 'amass':
            text =  random.choice(data['text'])['caption']
            face_token =  random.choice(face_token)
            hand_token =  random.choice(hand_token)
            lower_token =  random.choice(lower_token)
            upper_token =  random.choice(upper_token)
            emotion_label = "None"
        elif data_name.split('_')[0] == 'beat2':
            text = data['text']
            emotion_label = data['emotion_label']
        else:
            text = data['text']

            emotion_label = "None"


        tasks = data['tasks']

        # Convert data to tensors or set default values if data is None
        if upper_token is not None:
            m_tokens_len = upper_token.shape[0]
            face_token = torch.from_numpy(face_token).float()
            hand_token = torch.from_numpy(hand_token).float()
            lower_token = torch.from_numpy(lower_token).float()
            upper_token = torch.from_numpy(upper_token).float()
        else:
            m_tokens_len = 0
            face_token = torch.zeros(1)
            hand_token = torch.zeros(1)
            lower_token = torch.zeros(1)
            upper_token = torch.zeros(1)

        if audio_token is not None:
            a_tokens_len = audio_token.shape[0]
            audio_token = torch.from_numpy(audio_token).float()
        else:
            a_tokens_len = 0
            audio_token = torch.zeros(1)


        # Return the formatted sample
        return {
            "face_token": face_token,
            "hand_token": hand_token,
            "lower_token": lower_token,
            "upper_token": upper_token,
            "audio_token": audio_token,
            "tasks": tasks,
            "m_tokens_len": m_tokens_len,
            "a_tokens_len": a_tokens_len,
            "text": text,
            "split_name": 'train',
            "emotion_label": emotion_label
        }