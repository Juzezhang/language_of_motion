"""
This script is adapted from the original implementation found at:
https://github.com/OpenMotionLab/MotionGPT

Author: Biao Jiang and Xin Chen
Modified by: Juze Zhang
License: Check the original repository for licensing details.
"""
import os
from typing import List, Union, Dict, Any
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer, AutoModelForCausalLM
import random
from typing import Optional
from .tools.token_emb import NewTokenEmb


class MLM(nn.Module):

    def __init__(
        self,
        model_path: str,
        model_type: str = "t5",
        stage: str = "lm_pretrain",
        new_token_type: str = "insert",
        motion_codebook_size: int = 512,
        audio_codebook_size: int = 500,
        motion_framerate: float = 30.0,
        audio_samplerate: float = 16000.0,
        motion_down_sampling: int = 1,
        audio_down_sampling: int = 320,   ### audio down sample rate
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 512,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        flash_attention: bool = False,
        modalities: dict = None,
        **kwargs,
    ) -> None:
        super().__init__()

        # Parameters
        self.m_codebook_size = motion_codebook_size
        self.face_codebook_size = motion_codebook_size
        self.hand_codebook_size = motion_codebook_size
        self.upper_codebook_size = motion_codebook_size
        self.lower_codebook_size = motion_codebook_size
        self.a_codebook_size = audio_codebook_size
        self.max_length = max_length
        self.motion_framerate = motion_framerate
        self.audio_samplerate = audio_samplerate
        self.motion_down_sampling = motion_down_sampling
        self.audio_down_sampling = audio_down_sampling
        self.predict_ratio = predict_ratio
        self.inbetween_ratio = inbetween_ratio
        self.mask_ratio_audio = 0.08
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        self.quota_ratio = quota_ratio
        self.stage = stage

        # Instantiate language model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=True)
        if model_type == "t5":
            if flash_attention:
                from turbot5 import T5ForConditionalGeneration
                self.language_model = T5ForConditionalGeneration.from_pretrained(
                    model_path, attention_type = 'flash',use_triton=True)
            else:
                from transformers import T5ForConditionalGeneration
                self.language_model = T5ForConditionalGeneration.from_pretrained(
                    model_path)
        else:
            raise ValueError("model_type must be either t5, llama, or mistral")

        for modality, settings in modalities.items():
            prefix = settings["prefix"]
            codebook_size = settings["codebook_size"] + 3
            # Generate tokens for the current modality
            tokens = [f"<{prefix}_{i}>" for i in range(codebook_size)]
            self.tokenizer.add_tokens(tokens)

        self.language_model.resize_token_embeddings(len(self.tokenizer))


    def forward(self, 
                texts: List[str], 
                # text_timestamp: Optional[List[str]] = None,
                body_tokens: Optional[Dict[str, Tensor]] = None,
                audio_data: Optional[Dict[str, Tensor]] = None,
                lengths: Optional[Dict[str, List[int]]] = None,
                context: Optional[Dict[str, Any]] = None,
                tasks: Optional[dict] = None, 
                emotion_label: Optional[List[str]] = None):

        # Extract body tokens
        face_tokens = body_tokens.get('face') if body_tokens else None
        hand_tokens = body_tokens.get('hand') if body_tokens else None
        upper_tokens = body_tokens.get('upper') if body_tokens else None
        lower_tokens = body_tokens.get('lower') if body_tokens else None
        
        # Extract audio data
        audio_tokens = audio_data.get('tokens')
        # Extract length information
        motion_lengths = lengths.get('motion')
        audio_lengths = lengths.get('audio')
        # Extract context information
        emotion_label = context.get('emotion_label')

        face_strings, hand_strings, upper_strings, lower_strings, motion_string = self.compositional_motion_token_to_string(face_tokens, hand_tokens, upper_tokens, lower_tokens, motion_lengths)
        audio_strings = self.audio_token_to_string(audio_tokens, audio_lengths)
        inputs, outputs = self.template_fulfill(tasks, motion_lengths, audio_lengths, face_strings, hand_strings, upper_strings, lower_strings, motion_string, audio_strings, texts, emotion_label)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(face_tokens.device)
        source_input_ids = source_encoding.input_ids.to(face_tokens.device)

        target_inputs = self.tokenizer(outputs,
                                        padding='max_length',
                                        max_length=self.max_length,
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True,
                                        return_tensors="pt")

        labels_input_ids = target_inputs.input_ids.to(face_tokens.device)
        lables_attention_mask = target_inputs.attention_mask.to(
            face_tokens.device)

        labels_input_ids[labels_input_ids == 0] = -100
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask,
        )

        return outputs
    
    def generate_direct(self,
                        input: List[str] = None,
                        max_length: int = 512,
                        num_beams: int = 1,
                        do_sample: bool = True,
                        bad_words_ids: List[int] = None):

        # Device
        self.device = self.language_model.device

        # # Tokenize
        source_encoding = self.tokenizer(input,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        source_input_ids = source_encoding.input_ids.to(self.device)

        outputs = self.language_model.generate(
            source_input_ids,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=do_sample,
            bad_words_ids=bad_words_ids,
        )
        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)

        face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text = self.motion_string_to_compositional_token(outputs_string)
        return face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             body_tokens: Optional[Dict[str, Tensor]] = None,
                             audio_data: Optional[Dict[str, Tensor]] = None,
                             lengths: Optional[Dict[str, List[int]]] = None,
                             context: Optional[Dict[str, Any]] = None,
                             task: str = "t2m",
                             stage: str = 'train',
                             tasks: dict = None):
        """
        Generate motion tokens conditioned on various inputs with simplified parameter structure.
        
        Parameters:
        -----------
        texts: Optional[List[str]]
            Text prompts for text-to-motion generation
        
        body_tokens: Optional[Dict[str, Tensor]]
            Dictionary containing tokens for different body parts:
            {
                'face': face_tokens,
                'hand': hand_tokens,
                'upper': upper_tokens,
                'lower': lower_tokens
            }
        
        audio_data: Optional[Dict[str, Tensor]]
            Dictionary containing audio-related data:
            {
                'tokens': audio tokens,
                'onset': onset data,
                'amplitude_envelope': amplitude data,
                'timestamps': text timestamp alignment data
            }
            
        lengths: Optional[Dict[str, List[int]]]
            Dictionary containing length information:
            {
                'motion': motion sequence lengths,
                'audio': audio sequence lengths
            }
            
        context: Optional[Dict[str, Any]]
            Dictionary containing additional contextual information:
            {
                'combine_strings': combined textual representation,
                'emotion_label': emotion labels,
                'text_timestamp': timestamps for text
            }
            
        task: str
            Task type ('t2m', 'm2m', 'pred', 'inbetween', 'a2m', 'at2m', 'm2e', 'm2t', 'a2t')
            
        with_len: bool
            Whether to include length in the prompt
            
        stage: str
            Training stage
            
        tasks: dict
            Task specifications
        """
        self.device = self.language_model.device
        
        # Set default values for required dictionaries
        if context is None:
            context = {}
        if lengths is None:
            lengths = {}
        if audio_data is None:
            audio_data = {}
        
        # Extract body tokens
        face_tokens = body_tokens.get('face') if body_tokens else None
        hand_tokens = body_tokens.get('hand') if body_tokens else None
        upper_tokens = body_tokens.get('upper') if body_tokens else None
        lower_tokens = body_tokens.get('lower') if body_tokens else None
        
        # Extract audio data
        audio_tokens = audio_data.get('tokens')

        # Extract length information
        motion_lengths = lengths.get('motion')
        audio_lengths = lengths.get('audio')
        
        # Extract context information
        emotion_label = context.get('emotion_label')
        
        # Rest of the function implementation
        if task in ["t2m", "a2m"]:
            # Initialize string variables
            batch_size = 0
            # Determine batch size from available inputs
            if texts is not None:
                batch_size = len(texts)
            elif audio_tokens is not None:
                batch_size = len(audio_tokens)
            elif face_tokens is not None:
                batch_size = len(face_tokens)
            
            # Initialize empty strings for all inputs
            motion_strings = [''] * batch_size
            audio_strings = [''] * batch_size
            face_strings = [''] * batch_size
            hand_strings = [''] * batch_size
            upper_strings = [''] * batch_size
            lower_strings = [''] * batch_size
            combine_strings = [''] * batch_size
            emotion_strings = [''] * batch_size if emotion_label is None else emotion_label
            
            # Task-specific processing
            if task == "t2m":
                assert texts is not None, "Text input required for t2m task"
                audio_lengths = [0] * batch_size
                tasks = [{
                    'input': ['Generate motion: <Caption_Placeholder>'],
                    'output': ['']
                }] * batch_size
                lengths = [0] * batch_size
                
            elif task == "a2m":
                assert audio_tokens is not None, "Audio tokens required for a2m task"
                audio_strings = self.audio_token_to_string(audio_tokens, audio_lengths)
                tasks = [{
                    'input': ["Generate face motion: <AudioTranscript_Placeholder>"],
                    'output': ['']
                }] * batch_size
                lengths = [0] * batch_size
                
            # Create inputs and outputs from templates
            inputs, outputs = self.template_fulfill(
                tasks, lengths, audio_lengths,
                face_strings, hand_strings, upper_strings, lower_strings, 
                motion_strings, audio_strings, texts,
                combine_strings, emotion_strings
            )
            
            # Generate tokens using the language model
            face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text = self.generate_direct(
                inputs, max_length=self.max_length, num_beams=1, do_sample=True
            )
            
            # Return generated tokens as a dictionary for consistency
            return {
                'face': face_tokens,
                'hand': hand_tokens, 
                'upper': upper_tokens, 
                'lower': lower_tokens, 
                'text': cleaned_text
            }
        
    def compositional_motion_token_to_string(self, face_token: Tensor, hand_token: Tensor, upper_token: Tensor, lower_token: Tensor, lengths: List[int]):
        motion_string = []
        face_string = []
        hand_string = []
        upper_string = []
        lower_string = []

        # motion_string.append('<motion_id_0>')
        for i in range(len(lengths)):
            face_i = face_token[i].cpu() if face_token[i].device.type == 'cuda' else face_token[i]
            hand_i = hand_token[i].cpu() if hand_token[i].device.type == 'cuda' else hand_token[i]
            upper_i = upper_token[i].cpu() if upper_token[i].device.type == 'cuda' else upper_token[i]
            lower_i = lower_token[i].cpu() if lower_token[i].device.type == 'cuda' else lower_token[i]
            face_list = face_i.tolist()[:lengths[i]]
            hand_list = hand_i.tolist()[:lengths[i]]
            upper_list = upper_i.tolist()[:lengths[i]]
            lower_list = lower_i.tolist()[:lengths[i]]

            face_string_tmp = f'<face_id_{self.face_codebook_size}>'
            for j in range(lengths[i]):
                face_string_tmp = face_string_tmp + ''.join(f'<face_id_{int(face_list[j])}>')
            face_string_tmp += f'<face_id_{self.face_codebook_size+1}>'
            face_string.append(face_string_tmp)

            hand_string_tmp = f'<hand_id_{self.hand_codebook_size}>'
            for j in range(lengths[i]):
                hand_string_tmp = hand_string_tmp + ''.join(f'<hand_id_{int(hand_list[j])}>')
            hand_string_tmp += f'<hand_id_{self.hand_codebook_size+1}>'
            hand_string.append(hand_string_tmp)

            upper_string_tmp = f'<upper_id_{self.upper_codebook_size}>'
            for j in range(lengths[i]):
                upper_string_tmp = upper_string_tmp + ''.join(f'<upper_id_{int(upper_list[j])}>')
            upper_string_tmp += f'<upper_id_{self.upper_codebook_size+1}>'
            upper_string.append(upper_string_tmp)

            lower_string_tmp = f'<lower_id_{self.lower_codebook_size}>'
            for j in range(lengths[i]):
                lower_string_tmp = lower_string_tmp + ''.join(f'<lower_id_{int(lower_list[j])}>')
            lower_string_tmp += f'<lower_id_{self.lower_codebook_size+1}>'
            lower_string.append(lower_string_tmp)

            motion_string_tmp = '<motion_id_0>'
            for j in range(lengths[i]):
                motion_string_tmp = motion_string_tmp  + ''.join(f'<upper_id_{int(upper_list[j])}>') + ''.join(f'<lower_id_{int(lower_list[j])}>')
            motion_string_tmp += '<motion_id_1>'
            motion_string.append(motion_string_tmp)

        return face_string, hand_string, upper_string, lower_string, motion_string

    def audio_token_to_string(self, audio_token: Tensor, lengths: List[int]):
        audio_string = []
        for i in range(len(audio_token)):
            if audio_token[i] is None:
                continue
            audio_i = audio_token[i].cpu() if audio_token[i].device.type == 'cuda' else audio_token[i]
            audio_list = audio_i.tolist()[:lengths[i]]

            audio_string_tmp = f'<audio_id_{self.a_codebook_size}>'
            for j in range(lengths[i]):
                audio_string_tmp += ''.join(f'<audio_id_{int(audio_list[j])}>')
            audio_string_tmp += f'<audio_id_{self.a_codebook_size + 1}>'

            audio_string.append(audio_string_tmp)

        return audio_string

    def motion_token_list_to_string(self, motion_token: Tensor):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def motion_string_to_compositional_token(self, motion_string: List[str]):
        face_tokens = []
        hand_tokens = []
        lower_tokens = []
        upper_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str_emage(motion_string[i], '<motion_id_0>','<motion_id_1>')
            if string == '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>':

                face_string = self.get_middle_str_emage_v2(motion_string[i], f'<face_id_{self.face_codebook_size}>',f'<face_id_{self.face_codebook_size+1}>')
                hand_string = self.get_middle_str_emage_v2(motion_string[i], f'<hand_id_{self.hand_codebook_size}>',f'<hand_id_{self.hand_codebook_size+1}>')
                upper_string = self.get_middle_str_emage_v2(motion_string[i], f'<upper_id_{self.upper_codebook_size}>',f'<upper_id_{self.upper_codebook_size+1}>')
                lower_string = self.get_middle_str_emage_v2(motion_string[i], f'<lower_id_{self.lower_codebook_size}>',f'<lower_id_{self.lower_codebook_size+1}>')

                # string_list = string.split('><')
                face_string_list = face_string.split('><')
                hand_string_list = hand_string.split('><')
                upper_string_list = upper_string.split('><')
                lower_string_list = lower_string.split('><')

                face_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in face_string_list[1:-1] if i.startswith('face') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                hand_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in hand_string_list[1:-1] if i.startswith('hand') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                upper_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in upper_string_list[1:-1] if i.startswith('upper') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                lower_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in lower_string_list[1:-1] if i.startswith('lower') and i.split('_')[-1].replace('>', '').isdigit()
                ]

            else:
                string_list = string.split('><')
                face_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('face') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                hand_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('hand') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                lower_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('lower') and i.split('_')[-1].replace('>', '').isdigit()
                ]
                upper_token_list = [
                    int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if
                    i.startswith('upper') and i.split('_')[-1].replace('>', '').isdigit()
                ]


            if len(face_token_list) == 0:
                face_token_list = [0]
            if len(hand_token_list) == 0:
                hand_token_list = [0]
            if len(lower_token_list) == 0:
                lower_token_list = [0]
            if len(upper_token_list) == 0:
                upper_token_list = [0]

            face_token_list = torch.tensor(face_token_list, dtype=int).to(self.device)
            hand_token_list = torch.tensor(hand_token_list, dtype=int).to(self.device)
            lower_token_list = torch.tensor(lower_token_list, dtype=int).to(self.device)
            upper_token_list = torch.tensor(upper_token_list, dtype=int).to(self.device)

            face_tokens.append(face_token_list)
            hand_tokens.append(hand_token_list)
            lower_tokens.append(lower_token_list)
            upper_tokens.append(upper_token_list)

            if string == '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>':
                output_string.append(motion_string[i].replace(face_string, '<Face_Placeholder>')
                                     .replace(hand_string, '<Hand_Placeholder>')
                                     .replace(upper_string, '<Upper_Placeholder>')
                                     .replace(lower_string, '<Lower_Placeholder>'))
            else:
                output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))

        return face_tokens, hand_tokens, upper_tokens, lower_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, audio_length: int,
                                face_string: str, hand_string: str, upper_string: str,lower_string: str, motion_string: str,
                                audio_string: str, text: str, emotion_label: str):

        seconds = math.floor(length / self.motion_framerate)
        motion_splited = motion_string.split('>')
        face_splited = face_string.split('>')
        hand_splited = hand_string.split('>')
        upper_splited = upper_string.split('>')
        lower_splited = lower_string.split('>')
        audio_splited = audio_string.split('>')

        motion_token_length = length / self.motion_down_sampling

        # audio_token_length = audio_length / self.audio_down_sampling
        predict_head = int(motion_token_length * self.predict_ratio + 1)


        # Randomly choose the starting position and the length of the mask region
        mask_length = int(motion_token_length * self.inbetween_ratio)  # Calculate the length of the masked region
        start_index = random.randint(0,  int(motion_token_length - mask_length))  # Randomly select the starting index for masking
        # Ensure the mask region is within the bounds of the sequence
        masked_head = start_index  # The starting index of the masked region
        masked_tail = start_index + mask_length  # The ending index of the masked region

        mask_length_audio = int(audio_length * self.mask_ratio_audio)  # Calculate the length of the masked region
        start_index_audio = random.randint(0,  audio_length - mask_length_audio)  # Randomly select the starting index for masking
        masked_head_audio = start_index_audio  # The starting index of the masked region
        masked_tail_audio = start_index_audio + mask_length_audio  # The ending index of the masked region



        motion_predict_head = '>'.join(motion_splited[:predict_head]) + f'><motion_id_1>'
        motion_predict_last = f'<motion_id_0>' + '>'.join(motion_splited[predict_head:])
        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_2>' * (masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])

        face_predict_head = '>'.join(face_splited[:predict_head]) + f'><face_id_{self.face_codebook_size+1}>'
        face_predict_last = f'<face_id_{self.face_codebook_size}>' + '>'.join(face_splited[predict_head:])
        face_masked = '>'.join(
            face_splited[:masked_head]
        ) + '>' + f'<face_id_{self.face_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(face_splited[masked_tail:])

        hand_predict_head = '>'.join(hand_splited[:predict_head]) + f'><hand_id_{self.hand_codebook_size+1}>'
        hand_predict_last = f'<hand_id_{self.hand_codebook_size}>' + '>'.join(hand_splited[predict_head:])
        hand_masked = '>'.join(
            hand_splited[:masked_head]
        ) + '>' + f'<hand_id_{self.hand_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(hand_splited[masked_tail:])

        upper_predict_head = '>'.join(upper_splited[:predict_head]) + f'><upper_id_{self.upper_codebook_size+1}>'
        upper_predict_last = f'<upper_id_{self.upper_codebook_size}>' + '>'.join(upper_splited[predict_head:])
        upper_masked = ('>'.join(upper_splited[:masked_head]) + '>'
                        + f'<upper_id_{self.upper_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(upper_splited[masked_tail:]))

        lower_predict_head = '>'.join(lower_splited[:predict_head]) + f'><lower_id_{self.lower_codebook_size+1}>'
        lower_predict_last = f'<lower_id_{self.lower_codebook_size}>' + '>'.join(lower_splited[predict_head:])
        lower_masked = '>'.join(
            lower_splited[:masked_head]
        ) + '>' + f'<lower_id_{self.lower_codebook_size+2}>' * (masked_tail - masked_head) + '>'.join(lower_splited[masked_tail:])


        audio_masked = '>'.join(
            audio_splited[:masked_head_audio]
        ) + '>' + f'<audio_id_{self.a_codebook_size+2}>' * (masked_tail_audio - masked_head_audio) + '>'.join(audio_splited[masked_tail_audio:])


        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        if text == None:
            text = f'\"{text}\"'
        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Transcript_Placeholder>', text).replace(
            '<Emotion_Placeholder>', emotion_label).replace(
            '<Face_Placeholder>', face_string).replace(
            '<Hand_Placeholder>', hand_string).replace(
            '<Upper_Placeholder>', upper_string).replace(
            '<Lower_Placeholder>', lower_string).replace(
            '<Motion_Placeholder>', motion_string).replace(
            '<Audio_Placeholder>', audio_string).replace(
            '<Frame_Placeholder>',f'{length}').replace(
            '<Second_Placeholder>', '%.1f' % seconds).replace(
            '<Face_Placeholder_s1>', face_predict_head).replace(
            '<Hand_Placeholder_s1>', hand_predict_head).replace(
            '<Upper_Placeholder_s1>', upper_predict_head).replace(
            '<Lower_Placeholder_s1>', lower_predict_head).replace(
            '<Face_Placeholder_s2>', face_predict_last).replace(
            '<Hand_Placeholder_s2>', hand_predict_last).replace(
            '<Upper_Placeholder_s2>', upper_predict_last).replace(
            '<Lower_Placeholder_s2>', lower_predict_last).replace(
            '<Face_Placeholder_Masked>', face_masked).replace(
            '<Hand_Placeholder_Masked>', hand_masked).replace(
            '<Upper_Placeholder_Masked>', upper_masked).replace(
            '<Lower_Placeholder_Masked>', lower_masked).replace(
            '<Audio_Placeholder_Masked>', audio_masked)

        return prompt

    def template_fulfill(self,
                         tasks,
                         lengths,
                         audio_lengths,
                         face_strings,
                         hand_strings,
                         upper_strings,
                         lower_strings,
                         motion_string,
                         audio_strings,
                         texts,
                         emotion_label,
                         stage='test'):
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i], lower_strings[i], motion_string[i],
                                             audio_strings[i], texts[i], emotion_label[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i], lower_strings[i], motion_string[i],
                                             audio_strings[i], texts[i], emotion_label[i]))

        return inputs, outputs

    def get_middle_str(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
            if startIndex >= 0:
                startIndex += len(startStr)
            endIndex = content.index(endStr)
        except:
            return f'<motion_id_{self.m_codebook_size}><motion_id_0><motion_id_{self.m_codebook_size+1}>'

        return f'<motion_id_{self.m_codebook_size}>' + content[
            startIndex:endIndex] + f'<motion_id_{self.m_codebook_size+1}>'

    def get_middle_str_emage(self, content, startStr, endStr):

        try:
            startIndex = content.index(startStr)
        except:
            return '<motion_id_0><upper_id_0><lower_id_0><motion_id_1>'

        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return '<motion_id_0>' + content[startIndex:] + '<motion_id_1>'

        return '<motion_id_0>' + content[startIndex:endIndex] + '<motion_id_1>'

    def get_middle_str_emage_v2(self, content, startStr, endStr):
        try:
            startIndex = content.index(startStr)
        except:
            return startStr + '<face_id_0><hand_id_0><upper_id_0><lower_id_0>' + endStr
        
        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return startStr + content[startIndex:] + endStr

        return startStr + content[startIndex:endIndex] + endStr
