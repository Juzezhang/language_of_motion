import rich
import random
import pickle
import os
import numpy as np
import codecs as cs
from torch.utils import data
from os.path import join as pjoin
from rich.progress import track
import json
import spacy
from torchaudio.datasets import LIBRISPEECH
from os.path import join

class LibriSpeechDatasetEval(data.Dataset):
    def __init__(
        self,
        data_root,  # Root directory of the dataset
        token_root,  # Directory containing the tokenized data
        split,  # Dataset split, e.g., 'train' or 'test'
        tmpFile=True,  # Temporary file option
        tiny=False,  # If true, use a smaller dataset for debugging
        debug=False,  # Debug mode
        stage='lm_pretrain',  # Stage of training
        code_path='VQVAE',  # Path to VQ-VAE code
        used_testset='test-clean',  # Specifies the test dataset to use
        sample_rate=16000,  # Audio sample rate
        **kwargs,
    ):

        # Load pre-processed text data from a .pkl file
        with open(join(data_root, 'LibriSpeech', used_testset.split('_')[0] + '-texts.pkl'), 'rb') as f:
            self.text_dict = pickle.load(f)

        # Load pre-processed token data from a .pkl file
        with open(join(token_root, used_testset + '.pkl'), 'rb') as f:
            self.token_dict = pickle.load(f)

        # Preload all metadata (speaker ID, chapter ID, utterance ID)
        self.metadata = []
        for speaker_id in self.token_dict.keys():
            for chapter_id in self.token_dict[speaker_id].keys():
                for utterance_id in self.token_dict[speaker_id][chapter_id].keys():
                    self.metadata.append((speaker_id, chapter_id, utterance_id))


    def __len__(self):
        # Return the number of metadata entries
        return len(self.metadata)

    def __getitem__(self, item):
        # Get metadata for the specified item
        speaker_id, chapter_id, utterance_id = self.metadata[item]

        # Retrieve the corresponding token and text data
        a_token = self.token_dict[speaker_id][chapter_id][utterance_id]
        text = self.text_dict[speaker_id][chapter_id][utterance_id]

        # Get the length of the token data
        a_tokens_len = a_token.shape[0]

        # Return the text, tokens, and their length
        return text, a_token, a_tokens_len, None, None, None, None, None, None