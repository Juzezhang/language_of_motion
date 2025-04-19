import random
import numpy as np
from .dataset_a2m import Audio2MotionDataset
import torch


class Audio2MotionDatasetEval(Audio2MotionDataset):

    def __init__(
        self,
        # data_root,
        split,
        smpl_path,
        args,
        w_vectorizer,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        **kwargs,
    ):
        super().__init__(split,smpl_path, args, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer


    def __getitem__(self, item):
        # Get text data
        idx = item

        data = self.data_dict[self.name_list[idx]]
        # motion, m_length, audio_list,audio_length = data["motion"], data["length"], data["audio"], data["audio_length"]
        face, hand, lower, upper, tar_pose, tar_beta, tar_trans, tar_exps = data['face'], data['hand'], data['lower'], data['upper'], data['tar_pose'], data['tar_beta'], data['tar_trans'], data['tar_exps']

        raw_audio, audio_token = data['raw_audio'], data['audio']
        onset, amplitude_envelope= data['onset'], data['amplitude_envelope']

        m_tokens_len = torch.tensor(face.shape[0])
        a_tokens_len = audio_token.shape[0]

        # # Z Normalization
        # motion = (motion - self.mean) / self.std

        text_timestamp = data['text_timestamp']

        return {"face": face, "hand": hand, "lower": lower, "upper": upper, "tar_pose": tar_pose,
                "tar_beta": tar_beta, "tar_trans": tar_trans, "tar_exps": tar_exps ,"audio_token": audio_token,
                'onset': onset,
                'amplitude_envelope': amplitude_envelope,
                "raw_audio": raw_audio, "m_tokens_len": m_tokens_len, "a_tokens_len": a_tokens_len, "split_name": 'test', "text_timestamp": text_timestamp}