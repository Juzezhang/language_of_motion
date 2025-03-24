import rich
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
from loguru import logger

class Audio2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        args,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='TOKENS',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length
        self.training_speakers = args.training_speakers
        self.args = args
        self.ori_length = self.args.pose_length

        # Data path
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, code_path)
        # text_dir = pjoin(data_root, 'texts')
        audio_dir = pjoin(data_root, 'audios_token')

        split_rule = pd.read_csv(pjoin(data_root,"train_test_split.csv"))
        self.selected_file = split_rule.loc[(split_rule['type'] == split) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.training_speakers))]
        if args.additional_data and split == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])



        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr / 320)
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr


        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain':
            instructions = pjoin(data_root, 'template_pretrain.json')
        elif stage in ['lm_instruct', "lm_causal_instruct"]:
            instructions = pjoin(data_root, 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        # Data id list
        self.id_list = []
        for index, file_name in self.selected_file.iterrows():
            self.id_list.append(file_name["id"])


        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 100
            subset = '_tiny'
        else:
            enumerator = enumerate(
                track(
                    self.id_list,
                    f"Loading Beat2 {split}",
                ))
            maxdata = 1e10
            subset = ''

        # if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_tokens_data.pkl')):
        if False:
            if tiny or debug:
                with open(pjoin(data_root, f'tmp/{split}{subset}_tokens_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            else:
                with open(pjoin(data_root, f'tmp/{split}{subset}_tokens_data.pkl'),
                          'rb') as file:
                    data_dict = pickle.load(file)
            with open(pjoin(data_root, f'tmp/{split}{subset}_tokens_index.pkl'),
                      'rb') as file:
                new_name_list = pickle.load(file)
        else:

            new_name_list = []
            data_dict = {}

            # Fast loading
            for i, name in enumerator:
                if len(new_name_list) > maxdata:
                    break
                try:
                    # Load motion tokens
                    m_token_face = np.load(pjoin(motion_dir, 'face', f'{name}.npy'))
                    m_token_hands = np.load(pjoin(motion_dir, 'hands', f'{name}.npy'))
                    m_token_lower = np.load(pjoin(motion_dir, 'lower', f'{name}.npy'))
                    m_token_upper = np.load(pjoin(motion_dir, 'upper', f'{name}.npy'))
                    audio = np.load(pjoin(audio_dir, name + ".npy"))


                    motion_length = m_token_face.shape[1]
                    round_seconds_skeleton = motion_length // self.args.pose_fps  # assume 1500 frames / 15 fps = 100 s

                    clean_first_seconds = 0
                    clean_final_seconds = 0
                    clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds  # assume [10, 90]s
                    clip_s_f_audio, clip_e_f_audio = int(self.args.audio_fps * clip_s_t / 320 ), int(clip_e_t * self.args.audio_fps / 320 ) # [160,000,90*160,000]
                    clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps  # [150,90*15]

                    cut_length = int(self.ori_length)
                    audio_short_length = int(self.ori_length / self.args.pose_fps * self.args.audio_fps / 320 )

                    num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1

                    # logger.info(f"pose from frame {0} to {motion_length}, length {motion_length}")
                    # logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")

                    for clip_index in range(num_subdivision):  # cut into around 2s chip, (self npose)
                        start_idx = clip_s_f_pose + clip_index * self.args.stride
                        audio_start = clip_s_f_audio + math.floor(clip_index * self.args.stride * self.args.audio_fps/ 320 / self.args.pose_fps)

                        fin_idx = start_idx + cut_length
                        audio_end = audio_start + audio_short_length

                        sample_face = m_token_face[0, start_idx:fin_idx]
                        sample_hand = m_token_hands[0, start_idx:fin_idx]
                        sample_lower = m_token_lower[0, start_idx:fin_idx]
                        sample_upper = m_token_upper[0, start_idx:fin_idx]
                        sample_audio = audio[audio_start:audio_end]

                        data_dict[name + '_' + str(clip_index)] = {
                            'face_token': sample_face,
                            'hand_token': sample_hand,
                            'lower_token': sample_lower,
                            'upper_token': sample_upper,
                            # 'text': text_data
                            'audio': sample_audio
                        }
                        new_name_list.append(name + '_' + str(clip_index))


                except:
                    pass

            # if tmpFile:
            #     os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
            #     with open(
            #             pjoin(data_root,
            #                     f'tmp/{split}{subset}_tokens_data.pkl'),
            #             'wb') as file:
            #         pickle.dump(data_dict, file)
            #     with open(
            #             pjoin(data_root,
            #                     f'tmp/{split}{subset}_tokens_index.pkl'),
            #             'wb') as file:
            #         pickle.dump(new_name_list, file)

        self.data_dict = data_dict
        self.name_list = new_name_list
        self.nlp = spacy.load('en_core_web_sm')
        self.std_text = std_text
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):
        return len(self.name_list) * len(self.tasks)

    def __getitem__(self, item):
        # item = 300
        data_idx = item % len(self.name_list)
        task_idx = item // len(self.name_list)

        data = self.data_dict[self.name_list[data_idx]]
        face_token, hand_token, lower_token, upper_token, audio_token = data['face_token'], data['hand_token'], data['lower_token'], data['upper_token'], data['audio']

        tasks = self.tasks[task_idx]
        m_tokens_len = face_token.shape[0]
        a_tokens_len = audio_token.shape[0]


        face_token = torch.from_numpy(face_token).float()
        hand_token = torch.from_numpy(hand_token).float()
        lower_token = torch.from_numpy(lower_token).float()
        upper_token = torch.from_numpy(upper_token).float()
        audio_token = torch.from_numpy(audio_token).float()
        # m_tokens_len = torch.from_numpy(m_tokens_len)
        # a_tokens_len = torch.from_numpy(a_tokens_len)

        return {"face_token": face_token, "hand_token": hand_token, "lower_token": lower_token,
                "upper_token": upper_token, "audio_token": audio_token, "tasks": tasks, "m_tokens_len": m_tokens_len, "a_tokens_len": a_tokens_len}
