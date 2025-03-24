import numpy as np
import torch
import os 
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
from .mixed_dataset import MixedDatasetVQ, MixedDatasetCB, Audio2MotionDataset
from .utils import lom_collate


class MixedDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=lom_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'smplx'
        self.njoints = 55

        dataset_configs = cfg.DATASET.datasets
        # # Path to the dataset
        self.hparams.args = cfg.DATASET
        self.hparams.dataset_configs=dataset_configs
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE
        self.hparams.audio_down = cfg.model.params.lm.params.audio_down_sampling
        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        self.hparams.motion_representation = cfg.DATASET.motion_representation
        if self.hparams.motion_representation == "separate_rot" or self.hparams.motion_representation == "full_rot":
            self.hparams.smpl_path = cfg.DATASET.SMPL_PATH
            self.hparams.njoints = 55
        elif self.hparams.motion_representation == "h3d":
            self.hparams.njoints = 22
            dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m/t2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
            self.hparams.mean = np.load(pjoin(dis_data_root, "mean.npy"))
            self.hparams.std = np.load(pjoin(dis_data_root, "std.npy"))
            dis_data_root_eval = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m/t2m', "Comp_v6_KLD01", "meta")
            self.hparams.mean_eval = np.load(pjoin(dis_data_root_eval, "mean.npy"))
            self.hparams.std_eval = np.load(pjoin(dis_data_root_eval, "std.npy"))

        if cfg.TRAIN.STAGE == "vae" or cfg.TRAIN.STAGE == "vq" or cfg.TRAIN.STAGE == "token":
            self.Dataset = MixedDatasetVQ
            self.DatasetEval = MixedDatasetVQ
            # raise RuntimeError("Haven't setup this code!")
        elif 'lm' in cfg.TRAIN.STAGE:
            # Additional parameters
            # Length of the dataset
            # self.hparams.max_motion_length = cfg.DATASET.BEAT2.MAX_MOTION_LEN
            # self.hparams.min_motion_length = cfg.DATASET.BEAT2.MIN_MOTION_LEN
            # self.hparams.unit_length = cfg.DATASET.BEAT2.UNIT_LEN
            self.Dataset = MixedDatasetCB
            self.DatasetEval = Audio2MotionDataset
        # elif cfg.TRAIN.STAGE == "token":
        #     self.Dataset = MixedDatasetToken
        #     self.DatasetEval = MixedDatasetToken
        else:
            raise RuntimeError("Haven't setup this code!")

        # # Get additional info of the dataset
        # self._sample_set = self.get_sample_set(overrides={"split": "test", "tiny": True})


    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        example_data = np.load(os.path.join(self.hparams.data_root, 'joints', '000021.npy'))
        example_data = example_data.reshape(len(example_data), -1, 3)
        example_data = torch.from_numpy(example_data)
        features = process_file(features, self.njoints, example_data, 't2m')[0]
        return features

    def normalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = (features - mean) / std
        return features

    def denormalize(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return features

    def renorm4t2m(self, features):
        # renorm to t2m norms for using t2m evaluators
        ori_mean = torch.tensor(self.hparams.mean).to(features)
        ori_std = torch.tensor(self.hparams.std).to(features)
        eval_mean = torch.tensor(self.hparams.mean_eval).to(features)
        eval_std = torch.tensor(self.hparams.std_eval).to(features)
        features = features * ori_std + ori_mean
        features = (features - eval_mean) / eval_std
        return features

    def mm_mode(self, mm_on=True):
        if mm_on:
            self.is_mm = True
            self.name_list = self.test_dataset.name_list
            self.mm_list = np.random.choice(self.name_list,
                                            self.cfg.METRIC.MM_NUM_SAMPLES,
                                            replace=False)
            self.test_dataset.name_list = self.mm_list
        else:
            self.is_mm = False
            self.test_dataset.name_list = self.name_list

