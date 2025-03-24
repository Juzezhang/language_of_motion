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

class AudioText2MotionDatasetCB(data.Dataset):
    def __init__(
        self,
        data_root,
        split,
        mean,
        std,
        max_motion_length=196,
        min_motion_length=20,
        unit_length=4,
        fps=20,
        tmpFile=True,
        tiny=False,
        debug=False,
        stage='lm_pretrain',
        code_path='VQVAE',
        task_path=None,
        std_text=False,
        **kwargs,
    ):
        self.tiny = tiny
        self.unit_length = unit_length

        # Data mean and std
        self.mean = mean
        self.std = std

        # Data path
        split = 'train'
        split_file = pjoin(data_root, split + '.txt')
        motion_dir = pjoin(data_root, code_path)
        text_dir = pjoin(data_root, 'texts')
        audio_dir = pjoin(data_root, 'audios_token')

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
        with cs.open(split_file, "r") as f:
            for line in f.readlines():
                self.id_list.append(line.strip())

        # Debug mode
        if tiny or debug:
            # enumerator = enumerate(self.id_list)
            enumerator = enumerate(random.sample(self.id_list, len(self.id_list)))
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


        if os.path.exists(pjoin(data_root, f'tmp/{split}{subset}_tokens_data.pkl')):
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
                    m_token_list = np.load(pjoin(motion_dir, f'{name}.npy'))
                    if len(name) >= 10:
                        if name.startswith('M'):
                            audio = np.load(pjoin(audio_dir, name[1:] + ".npy"))[::2]
                        else:
                            audio = np.load(pjoin(audio_dir, name + ".npy"))[::2]
                        # Read text
                        text_data = {}
                        with cs.open(pjoin(text_dir, name + '.txt')) as f:
                            text_data['caption'] = f.readlines()[0]
                        data_dict[name] = {
                            'm_token_list': m_token_list,
                            'text': [text_data],
                            'audio': audio
                        }
                        new_name_list.append(name)

                    else:
                        # Load motion tokens
                        m_token_list = np.load(pjoin(motion_dir, f'{name}.npy'))
                        # Read text
                        with cs.open(pjoin(text_dir, name + '.txt')) as f:
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
                                        m_token_list_new = [
                                            tokens[int(f_tag * fps / unit_length
                                                       ):int(to_tag * fps /
                                                             unit_length)]
                                            for tokens in m_token_list
                                            if int(f_tag * fps / unit_length) <
                                               int(to_tag * fps / unit_length)
                                        ]

                                        if len(m_token_list_new) == 0:
                                            continue
                                        new_name = '%s_%f_%f' % (name, f_tag,
                                                                 to_tag)

                                        data_dict[new_name] = {
                                            'm_token_list': m_token_list_new,
                                            'text': [text_dict],
                                            'audio': None
                                        }
                                        new_name_list.append(new_name)
                                except:
                                    pass

                        if flag:
                            data_dict[name] = {
                                'm_token_list': m_token_list,
                                'text': text_data,
                                'audio': None
                            }
                            new_name_list.append(name)
                except:
                    pass

            if tmpFile:
                os.makedirs(pjoin(data_root, 'tmp'), exist_ok=True)
                with open(
                        pjoin(data_root,
                                f'tmp/{split}{subset}_tokens_data.pkl'),
                        'wb') as file:
                    pickle.dump(data_dict, file)
                with open(
                        pjoin(data_root,
                                f'tmp/{split}{subset}_tokens_index.pkl'),
                        'wb') as file:
                    pickle.dump(new_name_list, file)

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
        # task_idx = np.random.choice([0, 1, 2, 3])

        data = self.data_dict[self.name_list[data_idx]]
        m_token_list, text_list, audio_list = data['m_token_list'], data['text'], data['audio']

        m_tokens = random.choice(m_token_list)
        text_data = random.choice(text_list)
        caption = text_data['caption']
        if self.std_text:
            doc = self.nlp(caption)
            word_list = []
            pos_list = []
            for token in doc:
                word = token.text
                if not word.isalpha():
                    continue
                if (token.pos_ == 'NOUN'
                        or token.pos_ == 'VERB') and (word != 'left'):
                    word_list.append(token.lemma_)
                else:
                    word_list.append(word)
                pos_list.append(token.pos_)

            caption = ' '.join(word_list)

        # all_captions = [
        #     ' '.join([token.split('/')[0] for token in text_dic['tokens']])
        #     for text_dic in text_list
        # ]

        # coin = np.random.choice([False, False, True])
        #
        # if coin:
        #     # drop one token at the head or tail
        #     coin2 = np.random.choice([True, False])
        #     if coin2:
        #         m_tokens = m_tokens[:-1]
        #     else:
        #         m_tokens = m_tokens[1:]
        tasks = self.tasks[task_idx]
        m_tokens_len = m_tokens.shape[0]



        if audio_list is None:
            audio_list = np.array([0.])
            a_tokens_len = 0
        else:
            a_tokens_len = audio_list.shape[0]

        return caption, m_tokens, m_tokens_len, None, None, None, None, audio_list, a_tokens_len, tasks

