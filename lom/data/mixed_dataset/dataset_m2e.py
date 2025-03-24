import os
import rich
import random
import pickle
import codecs as cs
import numpy as np
from torch.utils import data
from rich.progress import track
from os.path import join as pjoin
import spacy
import pandas as pd
import math
from loguru import logger
# from utils_emage import rotation_conversions as rc
import torch
import smplx
from .data_tools import joints_list
import librosa
import textgrid as tg
import pandas as pd

class Motion2EmotionDataset(data.Dataset):

    def __init__(
        self,
        data_root,
        split,
        smpl_path,
        args,
        unit_length=1,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):

        smpl_device = torch.device("cpu")
        # # Set the cuda device
        # if torch.cuda.is_available():
        #     smpl_device = torch.device("cuda:0")
        #     torch.cuda.set_device(smpl_device)
        # else:
        #     smpl_device = torch.device("cpu")

        self.tiny = tiny
        self.unit_length = unit_length
        self.training_speakers = args.training_speakers
        self.args = args
        self.ori_length = self.args.pose_length
        self.ori_joint_list = joints_list[args.ori_joints]
        self.tar_joint_list = joints_list[args.tar_joints]
        self.smpl_path = smpl_path

        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1


        # Data path
        # split_file = pjoin(data_root, split + '.txt')
        # motion_dir = pjoin(data_root, self.args.pose_rep)
        motion_dir = pjoin(data_root, 'TOKENS')
        audio_dir = pjoin(data_root, 'audios_token')
        raw_audio_dir = pjoin(data_root, 'wave16k')

        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        # # select trainable joints
        # self.smplx_2020 = smplx.create(
        #     self.smpl_path,
        #     model_type='smplx',
        #     gender='NEUTRAL_2020',
        #     use_face_contour=False,
        #     num_betas=300,
        #     num_expression_coeffs=100,
        #     ext='npz',
        #     use_pca=False,
        # ).to(smpl_device).eval()

        split_rule = pd.read_csv(pjoin(data_root,"train_test_split.csv"))
        # self.selected_file = split_rule.loc[(split_rule['type'] == split) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.training_speakers))]
        self.selected_file = split_rule.loc[(split_rule['type'] == split) & (split_rule['id'].str.split("_").str[0].astype(int).isin([2]))]
        # self.selected_file = split_rule.loc[(split_rule['type'] == 'test') & (split_rule['id'].str.split("_").str[0].astype(int).isin([2]))]

        if args.additional_data and split == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])



        # Data id list
        self.id_list = []
        for index, file_name in self.selected_file.iterrows():
            self.id_list.append(file_name["id"])

        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr / 320)
        if self.max_audio_pre_len > self.args.test_length * self.args.audio_sr:
            self.max_audio_pre_len = self.args.test_length * self.args.audio_sr


        # Debug mode
        if tiny or debug:
            enumerator = enumerate(self.id_list)
            maxdata = 2
            # maxdata = 1e10
            subset = '_tiny'
        else:
            enumerator = enumerate(self.id_list)
            maxdata = 1e10
            # maxdata = 2
            subset = ''

        # maxdata = 1


        new_name_list = []
        length_list = []
        data_dict = {}


        for idx, name in enumerator:
            # if name != '2_scott_0_100_100':
            #     continue
            if len(new_name_list) > maxdata:
                break
            try:

                # Load motion tokens
                m_token_face = np.load(pjoin(motion_dir, 'face', f'{name}.npy'))
                m_token_hands = np.load(pjoin(motion_dir, 'hands', f'{name}.npy'))
                m_token_lower = np.load(pjoin(motion_dir, 'lower', f'{name}.npy'))
                m_token_upper = np.load(pjoin(motion_dir, 'upper', f'{name}.npy'))
                audio = np.load(pjoin(audio_dir, name + ".npy"))


                emotion_data = pd.read_csv(pjoin(data_root, 'emotion_label', name + ".csv"))
                emotion_label = emotion_data.columns[0].split('_')[-1]
                emotion_start_time = float(emotion_data.columns[1].split('_')[-1])
                emotion_stop_time = float(emotion_data.columns[2].split('_')[-1])

                audio_start =  int(emotion_start_time * 50.)
                audio_end = int(emotion_stop_time * 50.)
                motion_start =  int(emotion_start_time * 30.)
                motion_end = int(emotion_stop_time * 30.)

                data_dict[name] = {
                    'face_token': m_token_face[0, motion_start:motion_end],
                    'hand_token': m_token_hands[0, motion_start:motion_end],
                    'lower_token': m_token_lower[0, motion_start:motion_end],
                    'upper_token': m_token_upper[0, motion_start:motion_end],
                    # 'text': text_data
                    'audio': audio[audio_start:audio_end],
                    'emotion_label': emotion_label
                }
                new_name_list.append(name)
                length_list.append(motion_end - motion_start)

            except:
                pass

        name_list, length_list = zip(
            *sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

        # self.smplx_2020 = self.smplx_2020.to('cpu')
        torch.cuda.empty_cache()
        # self.nfeats = data_dict[name_list[0]]['motion'].shape[1]
        # self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)
        self.max_length = length

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, item):

        # idx = self.pointer + item
        idx = item

        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data["motion"], data["length"], data["text"]

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption = text_data["caption"]

        all_captions = [
            ' '.join([token.split('/')[0] for token in text_dic['tokens']])
            for text_dic in text_list
        ]

        # Crop the motions in to times of 4, and introduce small variations
        if self.unit_length < 10:
            coin2 = np.random.choice(["single", "single", "double"])
        else:
            coin2 = "single"

        if coin2 == "double":
            m_length = (m_length // self.unit_length - 1) * self.unit_length
        elif coin2 == "single":
            m_length = (m_length // self.unit_length) * self.unit_length

        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx + m_length]

        # Z Normalization
        motion = (motion - self.mean) / self.std

        return caption, motion, m_length, None, None, None, None, all_captions
