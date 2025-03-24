import os
import numpy as np
import pytorch_lightning as pl
import torch
from pathlib import Path
from tqdm import tqdm
from mGPT.config import parse_args
from mGPT.data.build_data import build_data
# from mGPT.models.build_model import build_model
# from mGPT.utils.load_checkpoint import load_pretrained_vae
from mGPT.archs.motion_representation import VQVAEConvZero, VAEConvZero
from utils_emage import other_tools
from mGPT.data.beat2.build_vocab import Vocab
from loguru import logger
from utils_emage import rotation_conversions as rc
from mGPT.data.beat2.data_tools import joints_list
import pandas as pd
from os.path import join as pjoin
import codecs as cs
import torch.nn.functional as F
import shutil


def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t

def main():
    # parse options
    cfg = parse_args(phase="test")  # parse config file
    cfg.TRAIN.STAGE = "token"
    cfg.TRAIN.BATCH_SIZE = 1

    # set seed
    pl.seed_everything(cfg.SEED_VALUE)

    # gpu setting
    if cfg.ACCELERATOR == "gpu":
        os.environ["PYTHONWARNINGS"] = "ignore"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"


    # Load each dataset based on its type from the configuration
    path_name_list = []
    dataset_name_list = []

    for config in cfg.DATASET.datasets:
        dataset_type = config.get("type")
        if dataset_type == "beat2":
            output_dir_beat2 = os.path.join(config.get("data_root"), 'reconstructed_motion')
            os.makedirs(output_dir_beat2, exist_ok=True)
            split_rule = pd.read_csv(pjoin(config.get("data_root"),"train_test_split.csv"))
            # selected_file = split_rule.loc[(split_rule['type'] == 'test' or split_rule['type'] == 'train' or split_rule['type'] == 'val')]
            selected_file = split_rule
            beat2_root = config.get("data_root")


    for index, file_name in selected_file.iterrows():

        f_name = file_name["id"]
        person_index = f_name.split('_')[0]
        path_old = pjoin("/nas/nas_32/AI-being/zhangjz/exp_motion/datasets/beat_english_v0.2.1/", person_index, f_name + '.csv')
        path_new = pjoin(beat2_root, 'emotion_label', f_name + '.csv')

        try:
            shutil.copy(path_old, path_new)
        except:
            pass


if __name__ == "__main__":
    main()
