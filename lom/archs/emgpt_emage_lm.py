import os
from typing import List, Union
import numpy as np
import math
import time
import heapq
import torch
from torch import Tensor, nn
from torch.distributions.distribution import Distribution
# from transformers import AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
# from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer, GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, AutoTokenizer
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
        framerate: float = 30.0,
        audio_samplerate: float = 16000.0,
        down_t: int = 1,
        down_a: int = 320,   ### audio down sample rate
        predict_ratio: float = 0.2,
        inbetween_ratio: float = 0.25,
        max_length: int = 512,
        lora: bool = False,
        quota_ratio: float = 0.5,
        noise_density: float = 0.15,
        mean_noise_span_length: int = 3,
        flash_attention: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        # Parameters
        # self.m_codebook_size = motion_codebook_size
        self.m_codebook_size = 256
        self.face_codebook_size = 256
        self.hand_codebook_size = 256
        self.upper_codebook_size = 256
        self.lower_codebook_size = 256

        self.a_codebook_size = audio_codebook_size
        self.max_length = max_length
        # self.max_length = 500
        self.framerate = framerate
        self.audio_samplerate = audio_samplerate
        self.down_t = down_t
        self.down_a = down_a
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

                # from .flashT5.src.model.modeling_flash_t5 import FlashT5ForConditionalGeneration
                # self.language_model = FlashT5ForConditionalGeneration.from_pretrained(
                #     model_path)
            else:
                from transformers import T5ForConditionalGeneration
                self.language_model = T5ForConditionalGeneration.from_pretrained(
                    model_path)
                # from transformers import T5ForConditionalGeneration, T5Config
                # config = T5Config.from_pretrained("t5-base")
                # self.language_model = T5ForConditionalGeneration(config)



            self.lm_type = 'encdec'
        # elif model_type == "gpt2":
        #     self.language_model = GPT2LMHeadModel.from_pretrained(model_path)
        #     self.lm_type = 'dec'
        else:
            raise ValueError("type must be either seq2seq or conditional")

        if self.lm_type == 'dec':
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add motion tokens
        self.tokenizer.add_tokens(
            [f'<motion_id_{i}>' for i in range(3)])
        # Add face tokens
        self.tokenizer.add_tokens(
            [f'<face_id_{i}>' for i in range(self.face_codebook_size+3)])
        # Add face tokens
        self.tokenizer.add_tokens(
            [f'<hand_id_{i}>' for i in range(self.hand_codebook_size+3)])
        # Add face tokens
        self.tokenizer.add_tokens(
            [f'<upper_id_{i}>' for i in range(self.upper_codebook_size+3)])
        # Add face tokens
        self.tokenizer.add_tokens(
            [f'<lower_id_{i}>' for i in range(self.lower_codebook_size+3)])
        # Add audio tokens
        self.tokenizer.add_tokens(
            [f'<audio_id_{i}>' for i in range(self.a_codebook_size + 5)])  ### pure audio we add 3, but we have audio_transcript further 2
            # [f'<audio_id_{i}>' for i in range(self.a_codebook_size + 3)])  ### pure audio we add 3, but we have audio_transcript further 2
        #
        # # Add motion tokens
        # self.tokenizer.add_tokens(
        #     [f'<motion_id_{i}>' for i in range(3)])
        # # Add face tokens
        # self.tokenizer.add_tokens(
        #     [f'<face_id_{i}>' for i in range(self.face_codebook_size)])
        # # Add face tokens
        # self.tokenizer.add_tokens(
        #     [f'<hand_id_{i}>' for i in range(self.hand_codebook_size)])
        # # Add face tokens
        # self.tokenizer.add_tokens(
        #     [f'<upper_id_{i}>' for i in range(self.upper_codebook_size)])
        # # Add face tokens
        # self.tokenizer.add_tokens(
        #     [f'<lower_id_{i}>' for i in range(self.lower_codebook_size)])
        # # Add audio tokens
        # self.tokenizer.add_tokens(
        #     [f'<audio_id_{i}>' for i in range(self.a_codebook_size + 3)])  ### pure audio we add 3, but we have audio_transcript further 2
        #     # [f'<audio_id_{i}>' for i in range(self.a_codebook_size + 3)])  ### pure audio we add 3, but we have audio_transcript further 2







        if new_token_type == "insert":
            self.language_model.resize_token_embeddings(len(self.tokenizer))
        elif new_token_type == "mlp":
            shared = NewTokenEmb(self.language_model.shared,
                                 self.m_codebook_size + 3)
            # lm_head = NewTokenEmb(self.language_model.lm_head,
            #   self.m_codebook_size + 3)
            self.language_model.resize_token_embeddings(len(self.tokenizer))
            self.language_model.shared = shared
            # self.language_model.lm_head = lm_head

        # Lora
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model, get_peft_model_state_dict
            from peft.utils.other import fsdp_auto_wrap_policy
            peft_config = LoraConfig(
                bias="none",
                task_type="CAUSAL_LM",
                #  inference_mode=False,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05)
            self.language_model = get_peft_model(self.language_model,
                                                 peft_config)

    def forward(self, texts: List[str], text_timestamp: List[str],face_token: Tensor, hand_token: Tensor, lower_token: Tensor, upper_token: Tensor, audio_tokens: Tensor,
                lengths: List[int], audio_length: List[int], tasks: dict, emotion_label:List[str]):
        if self.lm_type == 'encdec':
            return self.forward_encdec(texts, text_timestamp, face_token, hand_token, lower_token, upper_token, audio_tokens, lengths, audio_length, tasks,emotion_label)
        elif self.lm_type == 'dec':
            return self.forward_dec(texts, text_timestamp, face_token, hand_token, lower_token, upper_token, audio_tokens, lengths, audio_length, tasks,emotion_label)
        else:
            raise NotImplementedError("Only conditional_multitask supported")

    def forward_encdec(
        self,
        texts: List[str],
        text_timestamp: List[str],
        face_token: Tensor,
        hand_token: Tensor,
        lower_token: Tensor,
        upper_token: Tensor,
        audio_tokens: Tensor,
        lengths: List[int],
        audio_length: List[int],
        tasks: dict,
        emotion_label: List[str]
    ):

        # Tensor to string
        # motion_strings = self.emage_token_to_string(face_token, hand_token, lower_token, upper_token, lengths)
        # motion_strings = self.emage_token_to_string_v2(face_token, hand_token, lower_token, upper_token, lengths)
        face_strings, hand_strings, upper_strings, lower_strings, motion_string = self.emage_token_to_string_v3(face_token, hand_token, lower_token, upper_token, lengths)

        audio_strings = self.audio_token_to_string(audio_tokens, audio_length)
        combine_strings = self.audio_transcript_token_to_string(audio_tokens, text_timestamp, audio_length)

        # Supervised or unsupervised
        # condition = random.choice(
        #     ['text', 'motion', 'supervised', 'supervised', 'supervised'])
        condition = random.choice(['supervised', 'supervised', 'supervised'])

        if condition == 'text':
            inputs = texts
            outputs = texts
        elif condition == 'motion':
            inputs = face_strings + hand_strings + upper_strings + lower_strings
            outputs = face_strings + hand_strings + upper_strings + lower_strings
        else:
            inputs, outputs = self.template_fulfill_exp(tasks, lengths, audio_length, face_strings, hand_strings, upper_strings, lower_strings, motion_string, audio_strings, texts, combine_strings, emotion_label)

        # Tokenize
        source_encoding = self.tokenizer(inputs,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_attention_mask = source_encoding.attention_mask.to(face_token.device)
        source_input_ids = source_encoding.input_ids.to(face_token.device)

        if condition in ['text', 'motion']:
            batch_size, expandend_input_length = source_input_ids.shape
            mask_indices = np.asarray([
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ])
            target_mask = ~mask_indices
            input_ids_sentinel = self.create_sentinel_ids(
                mask_indices.astype(np.int8))
            target_sentinel = self.create_sentinel_ids(
                target_mask.astype(np.int8))

            labels_input_ids = self.filter_input_ids(source_input_ids,
                                                     target_sentinel)
            source_input_ids = self.filter_input_ids(source_input_ids,
                                                     input_ids_sentinel)

        else:
            target_inputs = self.tokenizer(outputs,
                                           padding='max_length',
                                           max_length=self.max_length,
                                           truncation=True,
                                           return_attention_mask=True,
                                           add_special_tokens=True,
                                           return_tensors="pt")

            labels_input_ids = target_inputs.input_ids.to(face_token.device)
            lables_attention_mask = target_inputs.attention_mask.to(
                face_token.device)

        labels_input_ids[labels_input_ids == 0] = -100
        outputs = self.language_model(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask if condition == 'supervised' else None,
            labels=labels_input_ids,
            decoder_attention_mask=lables_attention_mask if condition == 'supervised' else None,
        )

        return outputs

    def forward_dec(
        self,
        texts: List[str],
        motion_tokens: Tensor,
        audio_tokens: Tensor,
        lengths: List[int],
        tasks: dict,
    ):
        self.tokenizer.padding_side = "right"

        # Tensor to string
        motion_strings = self.motion_token_to_string(motion_tokens, lengths)

        # Supervised or unsupervised
        condition = random.choice(
            ['text', 'motion', 'supervised', 'supervised', 'supervised'])

        if condition == 'text':
            labels = texts
        elif condition == 'motion':
            labels = motion_strings
        else:
            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            labels = []
            for i in range(len(inputs)):
                labels.append(inputs[i] + ' \n ' + outputs[i] +
                              self.tokenizer.eos_token)

        # Tokenize
        inputs = self.tokenizer(labels,
                                padding='max_length',
                                max_length=self.max_length,
                                truncation=True,
                                return_attention_mask=True,
                                return_tensors="pt")

        labels_input_ids = inputs.input_ids.to(motion_tokens.device)
        lables_attention_mask = inputs.attention_mask.to(motion_tokens.device)
        outputs = self.language_model(input_ids=labels_input_ids,
                                      attention_mask=lables_attention_mask,
                                      labels=inputs["input_ids"])

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
        # if self.lm_type == 'dec':
        #     texts = [text + " \n " for text in texts]

        # source_encoding = self.tokenizer(texts,
        #                                  padding='max_length',
        #                                  max_length=self.max_length,
        #                                  truncation=True,
        #                                  return_attention_mask=True,
        #                                  add_special_tokens=True,
        #                                  return_tensors="pt")


        source_encoding = self.tokenizer(input,
                                         padding='max_length',
                                         max_length=self.max_length,
                                         truncation=True,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")
        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)

        if self.lm_type == 'encdec':
            outputs = self.language_model.generate(
                source_input_ids,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=do_sample,
                bad_words_ids=bad_words_ids,
            )
        elif self.lm_type == 'dec':
            outputs = self.language_model.generate(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=do_sample,
                max_new_tokens=max_length)
            self.tokenizer.padding_side = 'left'

        outputs_string = self.tokenizer.batch_decode(outputs,
                                                     skip_special_tokens=True)

        # outputs_tokens, cleaned_text = self.motion_string_to_token(
        #     outputs_string)
        # outputs_tokens, cleaned_text = self.emage_string_to_token(outputs_string)
        face_tokens, hand_tokens, upper_tokens, lower_tokens, cleaned_text = self.emage_string_to_token_seprate(outputs_string)
        return face_tokens, hand_tokens, lower_tokens, upper_tokens, cleaned_text

    def generate_conditional(self,
                             texts: Optional[List[str]] = None,
                             text_timestamp: Optional[List[str]] = None,
                             # motion_tokens: Optional[Tensor] = None,
                             face_tokens: Optional[Tensor] = None,
                             hand_tokens: Optional[Tensor] = None,
                             upper_tokens: Optional[Tensor] = None,
                             lower_tokens: Optional[Tensor] = None,
                             audio_tokens: Optional[Tensor] = None,
                             combine_strings: Optional[Tensor] = None,
                             lengths: Optional[List[int]] = None,
                             audio_lengths: Optional[List[int]] = None,
                             task: str = "t2m",
                             with_len: bool = False,
                             stage: str = 'train',
                             tasks: dict = None,
                             emotion_label: Optional[Tensor] = None,):

        self.device = self.language_model.device

        if task in ["t2m", "m2m", "pred", "inbetween", "a2m", "at2m"]:

            if task == "t2m":
                assert texts is not None
                motion_strings = [''] * len(texts)
                audio_strings = [''] * len(texts)
                face_strings = [''] * len(texts)
                hand_strings = [''] * len(texts)
                upper_strings = [''] * len(texts)
                lower_strings = [''] * len(texts)
                emotion_label = [''] * len(texts)
                audio_lengths = [0] * len(texts)
                combine_strings = [''] * len(texts)

                if not with_len:
                    if tasks is None:
                        tasks = [{
                            'input':
                            ['Generate motion: <Caption_Placeholder>'],
                            'output': ['']
                            # ['Generate upper and lower body motion: <Caption_Placeholder>'],
                            # 'output': ['']
                        }] * len(texts)

                    lengths = [0] * len(texts)
                else:
                    tasks = [{
                        'input': [
                            'Generate motion with <Frame_Placeholder> frames: <Caption_Placeholder>'
                        ],
                        'output': ['']
                    }] * len(texts)
            elif task == "a2m":
                assert audio_tokens is not None
                motion_strings = [''] * len(audio_tokens)
                face_strings = [''] * len(audio_tokens)
                hand_strings = [''] * len(audio_tokens)
                upper_strings = [''] * len(audio_tokens)
                lower_strings = [''] * len(audio_tokens)
                # combine_strings = [''] * len(audio_tokens)
                emotion_label = [''] * len(audio_tokens)
                audio_strings = self.audio_token_to_string(audio_tokens, audio_lengths)
                # audio_strings = [''] * len(audio_tokens)
                # combine_strings = self.audio_transcript_token_to_string(audio_tokens, text_timestamp, audio_lengths)
                combine_strings = [''] * len(audio_tokens)
                # tasks = [{
                #     'input':
                #     ["Generate motion: <Audio_Placeholder>"],
                #     'output': ['']
                # }] * len(audio_tokens)
                # tasks = [{
                #     'input':
                #     ["Generate face motion: <Audio_Placeholder>"],
                #     'output': ['']
                # }] * len(audio_tokens)
                tasks = [{
                    'input':
                    ["Generate face motion: <AudioTranscript_Placeholder>"],
                    'output': ['']
                }] * len(audio_tokens)
                lengths = [0] * len(audio_tokens)

            elif task == "at2m":
                assert audio_tokens is not None
                motion_strings = [''] * len(audio_tokens)
                face_strings = [''] * len(audio_tokens)
                hand_strings = [''] * len(audio_tokens)
                upper_strings = [''] * len(audio_tokens)
                lower_strings = [''] * len(audio_tokens)
                emotion_label = [''] * len(audio_tokens)
                audio_strings = [''] * len(audio_tokens)
                audio_lengths = [0] * len(texts)

                combine_strings = self.audio_transcript_token_to_string(audio_tokens, text_timestamp, audio_lengths)
                # tasks = [{
                #     'input':
                #     ["Generate motion: <AudioTranscript_Placeholder>"],
                #     'output': ['']
                # }] * len(audio_tokens)
                tasks = [{
                    'input':
                    ["Given the audio and transcript with precise timestamp alignment in \"<AudioTranscript_Placeholder>\", generate a coordinated motion sequence involving face, hand, upper, and lower body movements."],
                    'output': ['']
                }] * len(audio_tokens)
                lengths = [0] * len(audio_tokens)

            elif task == "pred":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)
                tasks = [{
                    'input': ['Predict motion: <Motion_Placeholder_s1>'],
                    'output': ['']
                }] * len(lengths)

                motion_strings_old = self.motion_token_to_string(
                    motion_tokens, lengths)
                motion_strings = []
                for i, length in enumerate(lengths):
                    split = length // 5
                    motion_strings.append(
                        '>'.join(motion_strings_old[i].split('>')[:split]) +
                        '>')

            elif task == "inbetween":
                assert motion_tokens is not None and lengths is not None
                texts = [''] * len(lengths)

                tasks = [{
                    'input': [
                        "Complete the masked motion: <Motion_Placeholder_Masked>"
                    ],
                    'output': ['']
                }] * len(lengths)
                motion_strings = self.motion_token_to_string(
                    motion_tokens, lengths)


            inputs, outputs = self.template_fulfill_exp(tasks, lengths, audio_lengths,
                                                        face_strings, hand_strings,
                                                        upper_strings, lower_strings,motion_strings,
                                                        audio_strings,texts,
                                                        combine_strings, emotion_label)


            face_tokens, hand_tokens, lower_tokens, upper_tokens, cleaned_text = self.generate_direct(inputs,
                                                                                                        max_length=self.max_length,
                                                                                                        num_beams=1,
                                                                                                        do_sample=True)


            return face_tokens, hand_tokens, lower_tokens, upper_tokens



        elif task == "m2e":

            face_strings, hand_strings, upper_strings, lower_strings, motion_strings = self.emage_token_to_string_v3(
                face_tokens, hand_tokens, lower_tokens, upper_tokens, lengths)
            audio_lengths = [0] * len(face_strings)
            audio_strings = [''] * len(face_strings)
            combine_strings = [''] * len(face_strings)
            texts = [''] * len(face_strings)

            tasks = [{
                'input':
                ["Identify the underlying emotion in the combined face, hand, upper, and lower body movements in <Upper_Placeholder><Lower_Placeholder>"],
                'output': ['']
            }] * len(face_strings)
            lengths = [0] * len(face_strings)

            inputs, outputs = self.template_fulfill_exp(tasks, lengths, audio_lengths,
                                                        face_strings, hand_strings,
                                                        upper_strings, lower_strings,motion_strings,
                                                        audio_strings,texts,
                                                        combine_strings, emotion_label)


            face_tokens, hand_tokens, lower_tokens, upper_tokens, cleaned_text = self.generate_direct(inputs,
                                                                                                        max_length=self.max_length,
                                                                                                        num_beams=1,
                                                                                                        do_sample=True)


            return face_tokens, hand_tokens, lower_tokens, upper_tokens, cleaned_text


        elif task == "m2t":
            assert motion_tokens is not None and lengths is not None

            motion_strings = self.motion_token_to_string(
                motion_tokens, lengths)

            if not with_len:
                tasks = [{
                    'input': ['Generate text: <Motion_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    motion_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )
            return cleaned_text

        elif task == "a2t":
            assert audio_tokens is not None and lengths is not None

            audio_strings = self.audio_token_to_string(
                audio_tokens, lengths)

            # if not with_len:
            #     tasks = [{
            #         'input': ['Generate text: <Audio_Placeholder>'],
            #         'output': ['']
            #     }] * len(lengths)
            if not with_len:
                tasks = [{
                    'input': ['<Audio_Placeholder>'],
                    'output': ['']
                }] * len(lengths)
            else:
                tasks = [{
                    'input': [
                        'Generate text with <Frame_Placeholder> frames: <Motion_Placeholder>'
                    ],
                    'output': ['']
                }] * len(lengths)

            texts = [''] * len(lengths)

            inputs, outputs = self.template_fulfill(tasks, lengths,
                                                    audio_strings, texts)
            outputs_tokens, cleaned_text = self.generate_direct(
                inputs,
                max_length=40,
                num_beams=1,
                do_sample=False,
                # bad_words_ids=self.bad_words_ids
            )

            return cleaned_text

    def motion_token_to_string(self, motion_token: Tensor, lengths: List[int]):
        motion_string = []
        for i in range(len(motion_token)):
            motion_i = motion_token[i].cpu(
            ) if motion_token[i].device.type == 'cuda' else motion_token[i]
            motion_list = motion_i.tolist()[:lengths[i]]
            motion_string.append(
                (f'<motion_id_{self.m_codebook_size}>' +
                 ''.join([f'<motion_id_{int(i)}>' for i in motion_list]) +
                 f'<motion_id_{self.m_codebook_size + 1}>'))
        return motion_string

    def emage_token_to_string(self, face_token: Tensor, hand_token: Tensor, lower_token: Tensor, upper_token: Tensor, lengths: List[int]):
        motion_string = []
        # motion_string.append('<motion_id_0>')
        for i in range(len(face_token)):
            face_i = face_token[i].cpu(
            ) if face_token[i].device.type == 'cuda' else face_token[i]
            face_list = face_i.tolist()[:lengths[i]]
            motion_string.append('<motion_id_0>' + ''.join([f'<face_id_{int(i)}>' for i in face_list]))

        for i in range(len(hand_token)):
            hand_i = hand_token[i].cpu(
            ) if hand_token[i].device.type == 'cuda' else hand_token[i]
            hand_list = hand_i.tolist()[:lengths[i]]
            motion_string[i] = motion_string[i] + ''.join([f'<hand_id_{int(i)}>' for i in hand_list])
        for i in range(len(lower_token)):
            lower_i = lower_token[i].cpu(
            ) if lower_token[i].device.type == 'cuda' else lower_token[i]
            lower_list = lower_i.tolist()[:lengths[i]]
            motion_string[i] = motion_string[i] + ''.join([f'<lower_id_{int(i)}>' for i in lower_list])
        for i in range(len(upper_token)):
            upper_i = upper_token[i].cpu(
            ) if upper_token[i].device.type == 'cuda' else upper_token[i]
            upper_list = upper_i.tolist()[:lengths[i]]
            motion_string[i] = motion_string[i] + ''.join([f'<upper_id_{int(i)}>' for i in upper_list]) + '<motion_id_1>'

        return motion_string


    def emage_token_to_string_v2(self, face_token: Tensor, hand_token: Tensor, lower_token: Tensor, upper_token: Tensor, lengths: List[int]):
        motion_string = []
        # motion_string.append('<motion_id_0>')
        for i in range(len(lengths)):
            face_i = face_token[i].cpu() if face_token[i].device.type == 'cuda' else face_token[i]
            hand_i = hand_token[i].cpu() if hand_token[i].device.type == 'cuda' else hand_token[i]
            lower_i = lower_token[i].cpu() if lower_token[i].device.type == 'cuda' else lower_token[i]
            upper_i = upper_token[i].cpu() if upper_token[i].device.type == 'cuda' else upper_token[i]
            face_list = face_i.tolist()[:lengths[i]]
            hand_list = hand_i.tolist()[:lengths[i]]
            lower_list = lower_i.tolist()[:lengths[i]]
            upper_list = upper_i.tolist()[:lengths[i]]
            motion_string_tmp = '<motion_id_0>'
            for j in range(lengths[i]):
                motion_string_tmp = motion_string_tmp + ''.join(f'<face_id_{int(face_list[j])}>') + ''.join(f'<hand_id_{int(hand_list[j])}>')  + ''.join(f'<lower_id_{int(lower_list[j])}>') + ''.join(f'<upper_id_{int(upper_list[j])}>')
            motion_string_tmp += '<motion_id_1>'
            motion_string.append(motion_string_tmp)

        return motion_string

    def emage_token_to_string_v3(self, face_token: Tensor, hand_token: Tensor, lower_token: Tensor, upper_token: Tensor, lengths: List[int]):
        motion_string = []
        face_string = []
        hand_string = []
        upper_string = []
        lower_string = []

        # motion_string.append('<motion_id_0>')
        for i in range(len(lengths)):
            face_i = face_token[i].cpu() if face_token[i].device.type == 'cuda' else face_token[i]
            hand_i = hand_token[i].cpu() if hand_token[i].device.type == 'cuda' else hand_token[i]
            lower_i = lower_token[i].cpu() if lower_token[i].device.type == 'cuda' else lower_token[i]
            upper_i = upper_token[i].cpu() if upper_token[i].device.type == 'cuda' else upper_token[i]
            face_list = face_i.tolist()[:lengths[i]]
            hand_list = hand_i.tolist()[:lengths[i]]
            lower_list = lower_i.tolist()[:lengths[i]]
            upper_list = upper_i.tolist()[:lengths[i]]

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

            # motion_string_tmp = '<motion_id_0>'
            # for j in range(lengths[i]):
            #     motion_string_tmp = motion_string_tmp + ''.join(f'<face_id_{int(face_list[j])}>') + ''.join(f'<hand_id_{int(hand_list[j])}>')  + ''.join(f'<lower_id_{int(lower_list[j])}>') + ''.join(f'<upper_id_{int(upper_list[j])}>')
            # motion_string_tmp += '<motion_id_1>'
            # motion_string.append(motion_string_tmp)

            motion_string_tmp = '<motion_id_0>'
            for j in range(lengths[i]):
                motion_string_tmp = motion_string_tmp  + ''.join(f'<lower_id_{int(lower_list[j])}>') + ''.join(f'<upper_id_{int(upper_list[j])}>')
            motion_string_tmp += '<motion_id_1>'
            motion_string.append(motion_string_tmp)


        return face_string, hand_string, upper_string, lower_string, motion_string

    def audio_transcript_token_to_string(self, audio_token: Tensor, text_timestamp: Tensor, lengths: List[int]):
        combined_string = []
        for i in range(len(audio_token)):
            if audio_token[i] is None:
                continue
            audio_i = audio_token[i].cpu() if audio_token[i].device.type == 'cuda' else audio_token[i]
            transcript_i = text_timestamp[i]

            audio_list = audio_i.tolist()[:lengths[i]]
            transcript_list = transcript_i[:lengths[i]]
            combined_string_tmp = f'<audio_id_{self.a_codebook_size + 3}>'
            for j in range(lengths[i]):
                combined_string_tmp = combined_string_tmp + ''.join(f'<audio_id_{int(audio_list[j])}>') + transcript_list[j]
            combined_string_tmp += f'<audio_id_{self.a_codebook_size + 4}>'
            combined_string.append(combined_string_tmp)

        return combined_string

    def audio_token_to_string(self, audio_token: Tensor, lengths: List[int]):
        audio_string = []
        for i in range(len(audio_token)):
            if audio_token[i] is None:
                continue
            audio_i = audio_token[i].cpu(
            ) if audio_token[i].device.type == 'cuda' else audio_token[i]
            audio_list = audio_i.tolist()[:lengths[i]]
            audio_string.append(
                (f'<audio_id_{self.a_codebook_size}>' +
                 ''.join([f'<audio_id_{int(i)}>' for i in audio_list]) +
                 f'<audio_id_{self.a_codebook_size + 1}>'))
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

    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<motion_id_{self.m_codebook_size}>',
                f'<motion_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            token_list = [
                int(i.split('_')[-1].replace('>', ''))
                for i in string_list[1:-1]
            ]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string

    def emage_string_to_token(self, motion_string: List[str]):
        face_tokens = []
        hand_tokens = []
        lower_tokens = []
        upper_tokens = []

        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str_emage(motion_string[i], '<motion_id_0>','<motion_id_1>')
            string_list = string.split('><')
            face_token_list = [
                int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if i.startswith('face') and i.split('_')[-1].replace('>', '').isdigit()
            ]
            hand_token_list = [
                int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if i.startswith('hand') and i.split('_')[-1].replace('>', '').isdigit()
            ]
            lower_token_list = [
                int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if i.startswith('lower') and i.split('_')[-1].replace('>', '').isdigit()
            ]
            upper_token_list = [
                int(i.split('_')[-1].replace('>', '')) for i in string_list[1:-1] if i.startswith('upper') and i.split('_')[-1].replace('>', '').isdigit()
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

            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return face_tokens, hand_tokens, lower_tokens, upper_tokens, output_string

    def emage_string_to_token_seprate(self, motion_string: List[str]):
        face_tokens = []
        hand_tokens = []
        lower_tokens = []
        upper_tokens = []

        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str_emage(motion_string[i], '<motion_id_0>','<motion_id_1>')

            # if string == '<motion_id_0><face_id_0><hand_id_0><lower_id_0><upper_id_0><motion_id_1>':
            if string == '<motion_id_0><lower_id_0><upper_id_0><motion_id_1>':

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

            # if string == '<motion_id_0><face_id_0><hand_id_0><lower_id_0><upper_id_0><motion_id_1>':
            if string == '<motion_id_0><lower_id_0><upper_id_0><motion_id_1>':

                output_string.append(motion_string[i].replace(face_string, '<Face_Placeholder>')
                                     .replace(hand_string, '<Hand_Placeholder>')
                                     .replace(upper_string, '<Upper_Placeholder>')
                                     .replace(lower_string, '<Lower_Placeholder>'))
            else:
                output_string.append(motion_string[i].replace(string, '<Motion_Placeholder>'))


        return face_tokens, hand_tokens, upper_tokens, lower_tokens, output_string


    def audio_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            string = self.get_middle_str(
                motion_string[i], f'<audio_id_{self.m_codebook_size}>',
                f'<audio_id_{self.m_codebook_size + 1}>')
            string_list = string.split('><')
            token_list = [
                int(i.split('_')[-1].replace('>', ''))
                for i in string_list[1:-1]
            ]
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Audio_Placeholder>'))

        return motion_tokens, output_string

    def placeholder_fulfill(self, prompt: str, length: int, audio_length: int, motion_string: str, audio_string: str,
                            text: str):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        motion_token_length = length / self.down_t
        # audio_token_length = audio_length / self.down_a
        predict_head = int(motion_token_length * self.predict_ratio + 1)
        masked_head = int(motion_token_length * self.inbetween_ratio + 1)
        masked_tail = int(motion_token_length * (1 - self.inbetween_ratio) + 1)
        
        motion_predict_head = '>'.join(
            motion_splited[:predict_head]
        ) + f'><motion_id_{self.m_codebook_size+1}>'
        motion_predict_last = f'<motion_id_{self.m_codebook_size}>' + '>'.join(
            motion_splited[predict_head:])

        motion_masked = '>'.join(
            motion_splited[:masked_head]
        ) + '>' + f'<motion_id_{self.m_codebook_size+2}>' * (
            masked_tail - masked_head) + '>'.join(motion_splited[masked_tail:])


        if random.random() < self.quota_ratio:
            text = f'\"{text}\"'

        if text == None:
            text = f'\"{text}\"'
        prompt = prompt.replace('<Caption_Placeholder>', text).replace(
            '<Motion_Placeholder>',motion_string).replace('<Audio_Placeholder>',
            audio_string).replace('<Frame_Placeholder>', f'{length}').replace(
                '<Second_Placeholder>', '%.1f' % seconds).replace(
                    '<Motion_Placeholder_s1>', motion_predict_head).replace(
                        '<Motion_Placeholder_s2>',
                        motion_predict_last).replace(
                            '<Motion_Placeholder_Masked>', motion_masked)

        return prompt

    def placeholder_fulfill_exp(self, prompt: str, length: int, audio_length: int,
                                face_string: str, hand_string: str, upper_string: str,lower_string: str, motion_string: str,
                                audio_string: str, text: str, combine_strings: str, emotion_label: str):

        seconds = math.floor(length / self.framerate)
        motion_splited = motion_string.split('>')
        face_splited = face_string.split('>')
        hand_splited = hand_string.split('>')
        upper_splited = upper_string.split('>')
        lower_splited = lower_string.split('>')
        audio_splited = audio_string.split('>')

        motion_token_length = length / self.down_t

        # audio_token_length = audio_length / self.down_a
        predict_head = int(motion_token_length * self.predict_ratio + 1)

        # masked_head = int(motion_token_length * self.inbetween_ratio + 1)
        # masked_tail = int(motion_token_length * (1 - self.inbetween_ratio) + 1)


        # Randomly choose the starting position and the length of the mask region
        mask_length = int(motion_token_length * self.inbetween_ratio)  # Calculate the length of the masked region
        start_index = random.randint(0,  motion_token_length - mask_length)  # Randomly select the starting index for masking
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
            '<AudioTranscript_Placeholder>', combine_strings).replace(
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
                         motion_strings,
                         audio_strings,
                         texts,
                         stage='test'):
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            # audio_lengths = [''] * len(lengths)
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i]
            inputs.append(
                self.placeholder_fulfill(input_template, length, audio_length,
                                         motion_strings[i], audio_strings[i], texts[i]))
            outputs.append(
                self.placeholder_fulfill(output_template, length, audio_length,
                                         motion_strings[i], audio_strings[i], texts[i]))

        return inputs, outputs

    def template_fulfill_exp(self,
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
                         combine_strings,
                         emotion_label,
                         stage='test'):
        inputs = []
        outputs = []
        if audio_lengths is None or audio_lengths[0] is None:
            # audio_lengths = [''] * len(lengths)
            audio_strings = [''] * len(lengths)

        for i in range(len(lengths)):
            input_template = random.choice(tasks[i]['input'])
            output_template = random.choice(tasks[i]['output'])
            length = lengths[i]
            audio_length = audio_lengths[i]
            inputs.append(
                self.placeholder_fulfill_exp(input_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i],lower_strings[i],motion_string[i],
                                             audio_strings[i], texts[i], combine_strings[i], emotion_label[i]))
            outputs.append(
                self.placeholder_fulfill_exp(output_template, length, audio_length,
                                             face_strings[i], hand_strings[i],
                                             upper_strings[i], lower_strings[i],motion_string[i],
                                             audio_strings[i], texts[i], combine_strings[i], emotion_label[i]))

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
        # try:
        #     startIndex = content.index(startStr)
        #     if startIndex >= 0:
        #         startIndex += len(startStr)
        #     endIndex = content.index(endStr)
        # except:
        #     return '<motion_id_0><face_id_0><hand_id_0><lower_id_0><upper_id_0><motion_id_1>'
        try:
            startIndex = content.index(startStr)
        except:
            # return '<motion_id_0><face_id_0><hand_id_0><lower_id_0><upper_id_0><motion_id_1>'
            return '<motion_id_0><lower_id_0><upper_id_0><motion_id_1>'

        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return '<motion_id_0>' + content[startIndex:] + '<motion_id_1>'

        return '<motion_id_0>' + content[startIndex:endIndex] + '<motion_id_1>'

    def get_middle_str_emage_v2(self, content, startStr, endStr):
        # try:
        #     startIndex = content.index(startStr)
        #     if startIndex >= 0:
        #         startIndex += len(startStr)
        #     endIndex = content.index(endStr)
        # except:
        #     return '<motion_id_0><face_id_0><hand_id_0><lower_id_0><upper_id_0><motion_id_1>'
        try:
            startIndex = content.index(startStr)
        except:
            return startStr + '<face_id_0><hand_id_0><lower_id_0><upper_id_0>' + endStr

        if startIndex >= 0:
            startIndex += len(startStr)
        try:
            endIndex = content.index(endStr)
        except:
            return startStr + content[startIndex:] + endStr

        return startStr + content[startIndex:endIndex] + endStr


    def random_spans_noise_mask(self, length):
        # From https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(
            np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens,
                                                  num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens,
                                                     num_noise_spans)

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length, ), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]

    def create_sentinel_ids(self, mask_indices):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        start_indices = mask_indices - np.roll(mask_indices, 1,
                                               axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(start_indices != 0,
                                np.cumsum(start_indices, axis=-1),
                                start_indices)
        sentinel_ids = np.where(sentinel_ids != 0,
                                (len(self.tokenizer) - sentinel_ids - (self.m_codebook_size + 3)), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        # From https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids,
                                  input_ids.to('cpu'))

        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape(
            (batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1),
                        self.tokenizer.eos_token_id,
                        dtype=np.int32),
            ],
            axis=-1,
        )

        input_ids = torch.tensor(input_ids, device=self.device)

        return input_ids
