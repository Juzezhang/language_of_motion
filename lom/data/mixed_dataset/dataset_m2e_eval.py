import random
import numpy as np
# from .dataset_a2m import Audio2MotionDataset
from .dataset_m2e import Motion2EmotionDataset

import torch


class Motion2EmotionDatasetEval(Motion2EmotionDataset):

    def __init__(
        self,
        data_root,
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
        super().__init__(data_root, split,smpl_path, args, unit_length, fps, tmpFile, tiny,
                         debug, **kwargs)

        self.w_vectorizer = w_vectorizer


    def __getitem__(self, item):
        # Get text data
        idx = item

        data = self.data_dict[self.name_list[idx]]
        # motion, m_length, audio_list,audio_length = data["motion"], data["length"], data["audio"], data["audio_length"]
        # face, hand, lower, upper, tar_pose, tar_beta, tar_trans, tar_exps = data['face'], data['hand'], data['lower'], data['upper'], data['tar_pose'], data['tar_beta'], data['tar_trans'], data['tar_exps']

        face_token, hand_token, lower_token, upper_token, audio_token = data['face_token'], data['hand_token'], data['lower_token'], data['upper_token'], data['audio']

        emotion_label = data['emotion_label'] + '/ADJ'

        word_emb, pos_oh = self.w_vectorizer[emotion_label]
        pos_one_hots = []
        word_embeddings = []
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)


        m_tokens_len = upper_token.shape[0]
        a_tokens_len = audio_token.shape[0]

        face_token = torch.from_numpy(face_token).float()
        hand_token = torch.from_numpy(hand_token).float()
        lower_token = torch.from_numpy(lower_token).float()
        upper_token = torch.from_numpy(upper_token).float()
        audio_token = torch.from_numpy(audio_token).float()
        sent_len = torch.from_numpy(np.array(1)).float()

        #
        # m_tokens_len = torch.tensor(face.shape[0])
        # a_tokens_len = audio_token.shape[0]
        #
        # # # Z Normalization
        # # motion = (motion - self.mean) / self.std
        #
        # text_timestamp = data['text_timestamp']

        return {
            "face_token": face_token,
            "hand_token": hand_token,
            "lower_token": lower_token,
            "upper_token": upper_token,
            "audio_token": audio_token,
            # "tasks": tasks,
            "m_tokens_len": m_tokens_len,
            "a_tokens_len": a_tokens_len,
            "split_name": 'test_m2e',
            "emotion_label": emotion_label,
            "pos_one_hots" : pos_one_hots,
            "word_embeddings" : word_embeddings,
            "sent_len": sent_len
        }