import numpy as np
import torch
from os.path import join as pjoin
from .humanml.utils.word_vectorizer import WordVectorizer
from .humanml.scripts.motion_process import (process_file, recover_from_ric)
from . import BASEDataModule
# from .humanml import Text2MotionDatasetEval, Text2MotionDataset, Text2MotionDatasetCB, MotionDataset, MotionDatasetVQ, Text2MotionDatasetToken, Text2MotionDatasetM2T, Audio2MotionDatasetCB, Audio2MotionDatasetEval
from .mixed_dataset import MixedDatasetToken, MixedDatasetCB, Audio2MotionDatasetEval
from .utils import beat2_emage_collate



class MixedDataModule(BASEDataModule):
    def __init__(self, cfg, **kwargs):

        super().__init__(collate_fn=beat2_emage_collate)
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        
        # Basic info of the dataset
        cfg.DATASET.JOINT_TYPE = 'beat2'
        self.name = "beat2"
        self.njoints = 55

        dataset_configs = cfg.DATASET.datasets
        # # Path to the dataset

        self.hparams.smpl_path =cfg.DATASET.smpl_path
        self.hparams.args = cfg.DATASET
        self.hparams.dataset_configs=dataset_configs
        self.hparams.debug = cfg.DEBUG
        self.hparams.stage = cfg.TRAIN.STAGE

        self.hparams.w_vectorizer = WordVectorizer(
            cfg.DATASET.WORD_VERTILIZER_PATH, "our_vab")

        # # Dataset switch
        data_root_beat2 = cfg.DATASET.BEAT2.ROOT
        self.hparams.data_root = data_root_beat2
        self.DatasetEval = Audio2MotionDatasetEval

        if cfg.TRAIN.STAGE == "vae":
            raise RuntimeError("Haven't setup this code!")
        elif 'lm' in cfg.TRAIN.STAGE:
            # Additional parameters
            self.hparams.debug = cfg.DEBUG
            self.hparams.stage = cfg.TRAIN.STAGE
            # Length of the dataset
            self.hparams.max_motion_length = cfg.DATASET.BEAT2.MAX_MOTION_LEN
            self.hparams.min_motion_length = cfg.DATASET.BEAT2.MIN_MOTION_LEN
            self.hparams.unit_length = cfg.DATASET.BEAT2.UNIT_LEN
            self.Dataset = MixedDatasetCB
        elif cfg.TRAIN.STAGE == "token":
            self.Dataset = MixedDatasetToken
            self.DatasetEval = MixedDatasetToken
        else:
            raise RuntimeError("Haven't setup this code!")

    def feats2joints(self, features):
        mean = torch.tensor(self.hparams.mean).to(features)
        std = torch.tensor(self.hparams.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.njoints)

    def joints2feats(self, features):
        features = process_file(features, self.njoints)[0]
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
