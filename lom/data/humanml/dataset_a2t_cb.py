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

class LibriSpeechDataset(data.Dataset):
    def __init__(
        self,
        data_root,  # Root directory of the dataset
        token_root,  # Directory containing tokenized data
        split,  # Dataset split, e.g., 'train' or 'test'
        tmpFile=True,  # Option to use a temporary file
        tiny=False,  # If true, use a smaller dataset for debugging
        debug=False,  # Debug mode
        stage='lm_pretrain',  # Stage of training
        code_path='VQVAE',  # Path to VQ-VAE code
        task_path=None,  # Path to task instructions
        used_trainset=['train-clean-100'],  # List of training datasets to use
        sample_rate=16000,  # Audio sample rate
        **kwargs,
    ):
        self.tiny = tiny
        self.datasets = []
        # Load datasets for each specified training set
        # for trainset in used_trainset:
        #     self.datasets.append(LIBRISPEECH(data_root, url=trainset, download=False))

        # Set the instructions file based on the training stage
        if task_path:
            instructions = task_path
        elif stage == 'lm_pretrain_audio':
            instructions = pjoin(data_root, 'LibriSpeech', 'template_pretrain.json')
        elif stage in ['lm_instruct', "lm_causal_instruct"]:
            instructions = pjoin(data_root, 'template_instructions.json')
        else:
            raise NotImplementedError(f"stage {stage} not implemented")

        # Pre-load all text and token data
        self.text_dict = {}
        self.token_dict = {}
        self.metadata = []

        for trainset in used_trainset:
            # with open(join(data_root, 'LibriSpeech', trainset.replace('_256','') + '-texts.pkl'), 'rb') as f:
            #     self.text_dict[trainset] = pickle.load(f)
            with open(join(data_root, 'LibriSpeech', trainset.split('_')[0] + '-texts.pkl'), 'rb') as f:
                self.text_dict[trainset] = pickle.load(f)

            with open(join(token_root, trainset + '.pkl'), 'rb') as f:
                self.token_dict[trainset] = pickle.load(f)

            # Collect metadata for each sample in the dataset
            for speaker_id in self.token_dict[trainset].keys():
                for chapter_id in self.token_dict[trainset][speaker_id].keys():
                    for utterance_id in self.token_dict[trainset][speaker_id][chapter_id].keys():
                        self.metadata.append((trainset, speaker_id, chapter_id, utterance_id))

        # Load NLP tools
        self.nlp = spacy.load('en_core_web_sm')

        # Load task instructions
        self.instructions = json.load(open(instructions, 'r'))
        self.tasks = []
        for task in self.instructions.keys():
            for subtask in self.instructions[task].keys():
                self.tasks.append(self.instructions[task][subtask])

    def __len__(self):
        # Return the total number of data samples, accounting for all tasks
        return len(self.metadata) * len(self.tasks)

    def __getitem__(self, item):
        # Determine the data and task index
        data_idx = item % len(self.metadata)
        task_idx = item // len(self.metadata)

        # Retrieve metadata for the current item
        trainset, speaker_id, chapter_id, utterance_id = self.metadata[data_idx]

        # Get the corresponding token and text data
        a_token = self.token_dict[trainset][speaker_id][chapter_id][utterance_id]
        text = self.text_dict[trainset][speaker_id][chapter_id][utterance_id]

        # Get the corresponding task
        tasks = self.tasks[task_idx]

        # Get the length of the token data
        a_tokens_len = a_token.shape[0]

        # Return the text, tokens, their length, and task information
        return text, a_token, a_tokens_len, None, None, None, None, None, tasks