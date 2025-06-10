import numpy as np
import os
import random
import torch
# import time
from lom.config import instantiate_from_config
from os.path import join as pjoin
from lom.losses.lom import GPTLosses
from lom.models.base import BaseModel
from lom.optimizers.loss_factory import get_loss_func
from lom.utils.rotation_conversions import rotation_6d_to_matrix, rotation_6d_to_axis_angle, matrix_to_axis_angle, matrix_to_rotation_6d, axis_angle_to_6d
from lom.utils.other_tools import velocity2position, estimate_linear_velocity
from .base import BaseModel
import json
from lom.data.mixed_dataset.data_tools import (
    joints_list, 
    JOINT_MASK_FACE,
    JOINT_MASK_UPPER,
    JOINT_MASK_HANDS,
    JOINT_MASK_LOWER,
    JOINT_MASK_FULL,
    BEAT_SMPLX_JOINTS,
    BEAT_SMPLX_FULL,
    BEAT_SMPLX_FACE,
    BEAT_SMPLX_UPPER,
    BEAT_SMPLX_HANDS,
    BEAT_SMPLX_LOWER,
    JOINT_MASK_JOINTS_6D,
    JOINT_MASK_FACE_6D,
    JOINT_MASK_HANDS_6D,
    JOINT_MASK_LOWER_6D,
    JOINT_MASK_UPPER_6D
)
import torch.nn.functional as F
import smplx
# from lom.models.utils.humanml3d_representation_converted import prepare_motion_representation_humanml3d
# vq_model_module = __import__(f"lom.archs.motion_representation", fromlist=["something"])


class Language_Motion(BaseModel):
    """
    Stage 1 Compositional Motion Tokenizer
    Stage 2 Pre-training for Modality Alignment
    Stage 3 Post-training with Instruction Following
    """

    def __init__(self,
                 cfg,
                #  datamodule,
                 lm,
                 modality_setup,
                 modality_tokenizer,
                 condition='text',
                 task='a2m',
                 metrics_dict=['CoSpeechMetrics'],
                 **kwargs):

        # self.save_hyperparameters(ignore='datamodule', logger=False)
        # self.datamodule = datamodule
        super().__init__()
        # self.args = cfg.DATASET
        self.vq_setting = cfg.vq
        self.audio_fps = modality_setup['params']['audio_fps']
        self.audio_down = modality_setup['params']['audio_down']
        self.motion_fps = modality_setup['params']['motion_fps']
        self.motion_down = modality_setup['params']['motion_down']
        self.test_length = cfg.TEST.TEST_LENGTH

        self.smplx = smplx.create(cfg.DATASET.SMPL_PATH,
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
            ).eval()
        
        # Instantiate modality tokenizer
        for tokenizer in modality_tokenizer:
            setattr(self, tokenizer, instantiate_from_config(modality_tokenizer[tokenizer]))

        # Instantiate motion-language model
        self.lm = instantiate_from_config(lm)
        # Get stage from lm config instead of hparams
        stage = getattr(self.lm, 'stage', None)
        if stage is not None:
            # Freeze the motion tokenizer for lm training
            if 'lm' in stage:
                self.vae_face.training = False
                for p in self.vae_face.parameters():
                    p.requires_grad = False
                self.vae_upper.training = False
                for p in self.vae_upper.parameters():
                    p.requires_grad = False
                self.vae_hand.training = False
                for p in self.vae_hand.parameters():
                    p.requires_grad = False
                self.vae_lower.training = False
                for p in self.vae_lower.parameters():
                    p.requires_grad = False
                self.vae_global.training = False
                for p in self.vae_global.parameters():
                    p.requires_grad = False

            if 'vae' in stage or 'vq' in stage:
                self.upper_loss = get_loss_func("UpperLoss")
                self.lower_loss = get_loss_func("LowerLoss")
                self.global_loss = get_loss_func("GlobalLoss")
                self.face_loss = get_loss_func("FaceLoss")
                self.hand_loss = get_loss_func("HandLoss")
                # Instantiate the losses

            config_losses = getattr(self.lm, 'losses', None)

            if config_losses is not None:
                # Instantiate the losses
                self._losses = torch.nn.ModuleDict({
                    split: GPTLosses(cfg, stage)
                    for split in ["losses_train", "losses_test", "losses_val"]
                })

        # # Data transform
        # self.feats2joints = datamodule.feats2joints
        # if hasattr(datamodule, 'feats2joints_exp'):
        #     self.feats2joints_exp = datamodule.feats2joints_exp


    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).to(filtered_t.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t.device)
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t

    def inverse_selection_tensor_full_body(self, filtered_t_face, filtered_t_hand, filtered_t_lower, filtered_t_upper, selection_array_face, selection_array_hand, selection_array_lower, selection_array_upper, n):
        selection_array_face = torch.from_numpy(selection_array_face).to(filtered_t_face.device)
        selection_array_hand = torch.from_numpy(selection_array_hand).to(filtered_t_hand.device)
        selection_array_lower = torch.from_numpy(selection_array_lower).to(filtered_t_lower.device)
        selection_array_upper = torch.from_numpy(selection_array_upper).to(filtered_t_upper.device)
        original_shape_t = torch.zeros((n, 165)).to(filtered_t_face.device)
        selected_indices_face = torch.where(selection_array_face == 1)[0]
        selected_indices_hand = torch.where(selection_array_hand == 1)[0]
        selected_indices_lower = torch.where(selection_array_lower == 1)[0]
        selected_indices_upper = torch.where(selection_array_upper == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices_face] = filtered_t_face[i] 
            original_shape_t[i, selected_indices_hand] = filtered_t_hand[i]
            original_shape_t[i, selected_indices_lower] = filtered_t_lower[i]
            original_shape_t[i, selected_indices_upper] = filtered_t_upper[i]
        return original_shape_t

    def inverse_selection_tensor_full_body_6D(self, filtered_t_face, filtered_t_hand, filtered_t_lower, filtered_t_upper, selection_array_face, selection_array_hand, selection_array_lower, selection_array_upper, n):
        selection_array_face = torch.from_numpy(selection_array_face).to(filtered_t_face.device)
        selection_array_hand = torch.from_numpy(selection_array_hand).to(filtered_t_hand.device)
        selection_array_lower = torch.from_numpy(selection_array_lower).to(filtered_t_lower.device)
        selection_array_upper = torch.from_numpy(selection_array_upper).to(filtered_t_upper.device)
        original_shape_t = torch.zeros((n, 330)).to(filtered_t_face.device)
        selected_indices_face = torch.where(selection_array_face == 1)[0]
        selected_indices_hand = torch.where(selection_array_hand == 1)[0]
        selected_indices_lower = torch.where(selection_array_lower == 1)[0]
        selected_indices_upper = torch.where(selection_array_upper == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices_face] = filtered_t_face[i] 
            original_shape_t[i, selected_indices_hand] = filtered_t_hand[i]
            original_shape_t[i, selected_indices_lower] = filtered_t_lower[i]
            original_shape_t[i, selected_indices_upper] = filtered_t_upper[i]
        return original_shape_t
    
    def pad_tensor(self, tensor, target_length):
        """Pad the given tensor to the target length or truncate it to the target length."""
        current_length = tensor.size(0)

        if current_length < target_length:
            # Pad with zeros to the target length, ensuring the padding tensor is on the same device as the original tensor
            padding_length = target_length - current_length
            return torch.cat([tensor, torch.zeros(padding_length, device=tensor.device)], dim=0)
        elif current_length > target_length:
            # If the length exceeds the target length, truncate it
            return tensor[:target_length]
        return tensor

    def unify_length(self, outputs_face, outputs_hand, outputs_lower, outputs_upper, motion_length):
        """Unify the length of all tensors in all lists to the maximum length among the four lists."""

        max_length = motion_length
        # Unify the length of each tensor in the four lists
        outputs_face = [self.pad_tensor(tensor, max_length) for tensor in outputs_face]
        outputs_hand = [self.pad_tensor(tensor, max_length) for tensor in outputs_hand]
        outputs_lower = [self.pad_tensor(tensor, max_length) for tensor in outputs_lower]
        outputs_upper = [self.pad_tensor(tensor, max_length) for tensor in outputs_upper]

        return outputs_face, outputs_hand, outputs_lower, outputs_upper

    def forward(self, batch, task="t2m"):

        if "audio" in batch:
            audio_token_string = batch["audio"]
            audio_token_string = [f"Based on {audio}, generate a synchronized movement sequence involving both upper, lower, face and hands body." for audio in audio_token_string]
            outputs_face, outputs_hand, outputs_upper, outputs_lower, output_texts = self.lm.generate_direct(input=audio_token_string, do_sample=True)
            feats_face, feats_hand, feats_upper, feats_lower = self.unify_length(outputs_face, outputs_hand, outputs_upper, outputs_lower, self.test_length)

        if "text" in batch:
            text_token_string = batch["text"]
            # text_token_string = ['Generate upper and lower body motion from given caption: ' + text for text in text_token_string]
            outputs_face, outputs_hand, outputs_upper, outputs_lower, output_texts = self.lm.generate_direct(input=text_token_string, do_sample=True)
            output_length = max(len(outputs_face[0]), len(outputs_hand[0]), len(outputs_upper[0]), len(outputs_lower[0]))
            feats_face = torch.zeros([len(outputs_face), output_length]).to(self.device)
            feats_hand = torch.zeros([len(outputs_hand), output_length]).to(self.device)
            feats_upper = torch.zeros([len(outputs_upper), output_length]).to(self.device)
            feats_lower = torch.zeros([len(outputs_lower), output_length]).to(self.device)

            # padding and concat
            for i in range(len(feats_face)):
                if outputs_face[i].shape[0] <= output_length:
                    feats_face[i, :outputs_face[i].shape[0]] = outputs_face[i]
                else:
                    feats_face[i, :output_length] = outputs_face[i][:output_length]
            # padding and concat
            for i in range(len(feats_hand)):
                if outputs_hand[i].shape[0] <= output_length:
                    feats_hand[i, :outputs_hand[i].shape[0]] = outputs_hand[i]
                else:
                    feats_hand[i, :output_length] = outputs_hand[i][:output_length]
            # padding and concat
            for i in range(len(feats_upper)):
                if outputs_upper[i].shape[0] <= output_length:
                    feats_upper[i, :outputs_upper[i].shape[0]] = outputs_upper[i]
                else:
                    feats_upper[i, :output_length] = outputs_upper[i][:output_length]
            # padding and concat
            for i in range(len(feats_lower)):
                if outputs_lower[i].shape[0] <= output_length:
                    feats_lower[i, :outputs_lower[i].shape[0]] = outputs_lower[i]
                else:
                    feats_lower[i, :output_length] = outputs_lower[i][:output_length]

        outputs = {
            "face": feats_face,
            "hand": feats_hand,
            "upper": feats_upper,
            "lower": feats_lower
        }

        return outputs

    def train_lm_forward(self, batch):

        body_tokens = {
            'face':  batch['face_token'],
            'hand': batch['hand_token'],
            'upper': batch['upper_token'],
            'lower': batch['lower_token']
        }

        lengths_motion = batch["m_tokens_len"]
        emotion_label = batch["emotion_label"]

        batch_size = body_tokens['face'].shape[0]

        if "text" in batch:
            texts = batch["text"]
        else:
            texts = [None] * batch_size

        if 'audio_token' in batch:
            audio_tokens = batch['audio_token']
            audio_length = batch['a_tokens_len']
        else:
            audio_tokens = [None] * batch_size
            audio_length = [None] * batch_size
        tasks = batch["tasks"]

        context = {
            'emotion_label': emotion_label,
        }
        lengths={
            'motion': lengths_motion,
            'audio': audio_length
        }
        # LLM Forward
        outputs = self.lm(texts, 
                          body_tokens = body_tokens,
                          audio_data = {'tokens': audio_tokens},
                          lengths=lengths,
                          context=context,
                          tasks=tasks,
                          emotion_label=emotion_label)

        return {'outputs': outputs}


    @torch.no_grad()
    def val_t2m_forward(self, batch):

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        tasks = None
        if self.trainer.datamodule.is_mm:
            texts = texts * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS
            instructions = pjoin(self.datamodule.hparams.data_root,
                                 'template_instructions.json')
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["caption"]] * len(texts)

        if self.hparams.condition == 'caption':
            tasks = [{
                'input': ['<Caption_Placeholder>'],
                'output': ['']
            }] * len(texts)

        if self.hparams.cfg.DATASET.TASK_PATH:
            instructions = pjoin(self.hparams.cfg.DATASET.TASK_PATH)
            instructions = json.load(open(instructions, 'r'))
            tasks = [instructions["Text-to-Motion"]["t2m"]] * len(texts)



        face_tokens, hand_tokens, lower_tokens, upper_tokens = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        self.vq_model_face = self.vq_model_face.to(torch.float)
        self.vq_model_upper = self.vq_model_upper.to(torch.float)
        self.vq_model_lower = self.vq_model_lower.to(torch.float)
        self.vq_model_hands = self.vq_model_hands.to(torch.float)


        feats_rst = torch.zeros_like(feats_ref)

        min_len = lengths.copy()
        j = 55
        motion_device = feats_ref.device

        for i in range(len(texts)):

            # output_length = max(len(face_tokens[i]), len(hand_tokens[i]), len(lower_tokens[i]), len(upper_tokens[i]))
            output_length = min(len(lower_tokens[i]), len(upper_tokens[i]))
            feats_face, feats_hand, feats_lower, feats_upper = self.unify_length(face_tokens[i:i+1], hand_tokens[i:i+1], lower_tokens[i:i+1], upper_tokens[i:i+1], output_length)

            feats_face = torch.stack(feats_face, dim=0)
            feats_hand = torch.stack(feats_hand, dim=0)
            feats_upper = torch.stack(feats_upper, dim=0)
            feats_lower = torch.stack(feats_lower, dim=0)

            feats_face = torch.clamp(feats_face,0, self.lm.face_codebook_size - 1, out=None)
            feats_hand = torch.clamp(feats_hand,0, self.lm.hand_codebook_size - 1, out=None)
            feats_upper = torch.clamp(feats_upper,0, self.lm.upper_codebook_size - 1, out=None)
            feats_lower = torch.clamp(feats_lower,0, self.lm.lower_codebook_size - 1, out=None)

            rec_face = self.vq_model_face.decode(feats_face.int())
            rec_hands = self.vq_model_hands.decode(feats_hand.int())
            rec_upper = self.vq_model_upper.decode(feats_upper.int())
            rec_lower = self.vq_model_lower.decode(feats_lower.int())

            rec_face = rec_face.float()
            rec_exps = rec_face[:, :, 6:]
            rec_pose_jaw = rec_face[:, :, :6]
            rec_pose_legs = rec_lower[:, :, :54]
            bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
            rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
            rec_pose_upper = rotation_6d_to_matrix(rec_pose_upper)  #
            rec_pose_upper = matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper.cuda(), JOINT_MASK_UPPER, bs*n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
            rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
            rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower.cuda(), JOINT_MASK_LOWER, bs * n)
            rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
            rec_pose_hands = rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands.cuda(), JOINT_MASK_HANDS, bs * n)
            rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
            rec_pose_jaw = rotation_6d_to_matrix(rec_pose_jaw)
            rec_pose_jaw = matrix_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
            rec_pose[:, 66:69] = rec_pose_jaw

            to_global = rec_lower
            if to_global.shape[2] == 54:
                to_global = F.pad(to_global, (0, 7))
            to_global[:, :, 54:57] = 0.0
            to_global[:, :, :54] = rec_lower2global
            rec_global = self.global_motion(to_global)

            rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
            rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, torch.zeros(rec_trans_v_s[:, 0, 0:1].shape).to(motion_device))
            rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, torch.zeros(rec_trans_v_s[:, 0, 2:3].shape).to(motion_device))
            rec_y_trans = rec_trans_v_s[:, :, 1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)

            rec_beta = torch.zeros(n, 300).to(motion_device)
            vertices_rec = self.smplx(
                betas=rec_beta.reshape(n, 300),
                transl=rec_trans[0].reshape(n, 3),
                expression=rec_exps[0].reshape(n, 100),
                jaw_pose=rec_pose[:, 66:69],
                global_orient=rec_pose[:, :3],
                body_pose=rec_pose[:, 3:21 * 3 + 3],
                left_hand_pose=rec_pose[:, 25 * 3:40 * 3],
                right_hand_pose=rec_pose[:, 40 * 3:55 * 3],
                return_joints=True,
                leye_pose=rec_pose[:, 69:72],
                reye_pose=rec_pose[:, 72:75],
            )

            # pose_seq_np = body.Jtr.detach().cpu().numpy()
            joints_rec = vertices_rec["joints"].float().detach().cpu().numpy().reshape(1, n, 127 * 3)[0, :, :55 * 3]

            new_joints, new_joints_vecs = prepare_motion_representation_humanml3d(joints_rec)

            output_length_ds =new_joints_vecs.shape[0]
            min_len[i] = min(output_length_ds, lengths[i])
            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = torch.from_numpy(new_joints_vecs[:lengths[i]]).unsqueeze(0)


        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        # joints_rst = self.feats2joints(feats_rst)
        joints_rst = self.feats2joints_exp(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        # feats_rst = self.datamodule.renorm4t2m(feats_rst)
        feats_rst = self.datamodule.renorm4t2m_exp(feats_rst)

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "m_rst": feats_rst,
            "joints_ref": joints_ref,
            "joints_rst": joints_rst,
            "length": min_len
            # "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_a2m_forward(self, batch):
        # Extract data from batch
        face = batch['face']
        hand = batch['hand']
        lower = batch['lower']
        upper = batch['upper']
        tar_pose_6d = batch["tar_pose"]
        tar_beta = batch["tar_beta"]
        tar_trans = batch["tar_trans"]
        tar_exps = batch["tar_exps"]
        lengths = batch["m_tokens_len"]
        raw_audio = batch["raw_audio"]
        batch_size = face.shape[0]

        # Handle text inputs
        if "text" in batch:
            texts = batch["text"]
        else:
            texts = [None] * batch_size
        
        # Handle audio inputs
        if 'audio_token' in batch:
            audio_tokens = batch['audio_token']
            audio_length = batch['a_tokens_len']
            text_timestamp = batch['text_timestamp']
            max_audio_length = max(audio_length)
            text_timestamp = [sublist + ['None'] * (max_audio_length - len(sublist)) for sublist in text_timestamp]
        else:
            audio_tokens = [None] * batch_size
            audio_length = [None] * batch_size

        bs, n = tar_pose_6d.shape[0], tar_pose_6d.shape[1]

        # Handle remainder frames
        j = 55
        remain = n % 8
        if remain != 0:
            tar_pose_6d = tar_pose_6d[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            face = face[:, :-remain, :]
            hand = hand[:, :-remain, :]
            lower = lower[:, :-remain, :]
            upper = upper[:, :-remain, :]
            n = n - remain


        audio_token_fps = self.audio_fps / self.audio_down
        motion_token_fps = self.motion_fps / self.motion_down

        # Calculate rounds and remaining frames
        roundt = (n - self.args.pre_frames) // (self.args.test_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.test_length - self.args.pre_frames)
        round_l = self.args.test_length - self.args.pre_frames

        # Initialize containers for token indices
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []

        texts = [None] * bs
        tasks = None
        
        # Process audio in chunks
        for i in range(0, roundt):
            # Calculate audio chunk indices
            start_idx = i * int(audio_token_fps / motion_token_fps * round_l)
            end_idx = (i + 1) * int(audio_token_fps / motion_token_fps * round_l) + int(
                audio_token_fps / motion_token_fps * self.args.pre_frames)

            # Get audio chunk
            audio_tokens_tmp = audio_tokens[:, start_idx:end_idx]
            text_timestamp_tmp = [sublist[start_idx:end_idx] for sublist in text_timestamp]

            # Use the new structured parameter approach for generate_conditional
            result = self.lm.generate_conditional(
                texts=texts,
                audio_data={
                    'tokens': audio_tokens_tmp,
                    'timestamps': text_timestamp_tmp
                },
                lengths={
                    'motion': lengths,
                    'audio': [len(audio_token) for audio_token in audio_tokens_tmp]
                },
                task='a2m',
                stage='test',
                tasks=tasks
            )
            
            # Extract results from the dictionary return value
            outputs_face = result['face']
            outputs_hand = result['hand']
            outputs_lower = result['lower']
            outputs_upper = result['upper']

            # Unify tensor lengths
            feats_face, feats_hand, feats_lower, feats_upper = self.unify_length(
                outputs_face, outputs_hand, outputs_lower, outputs_upper, int(self.args.test_length/self.motion_down))

            # Stack tensors
            feats_face = torch.stack(feats_face, dim=0)
            feats_hand = torch.stack(feats_hand, dim=0)
            feats_lower = torch.stack(feats_lower, dim=0)
            feats_upper = torch.stack(feats_upper, dim=0)

            # Collect results, handling the first chunk differently
            if i == 0:
                rec_index_all_face.append(feats_face)
                rec_index_all_upper.append(feats_upper)
                rec_index_all_lower.append(feats_lower)
                rec_index_all_hands.append(feats_hand)
            else:
                rec_index_all_face.append(feats_face[:, self.args.pre_frames:])
                rec_index_all_upper.append(feats_upper[:, self.args.pre_frames:])
                rec_index_all_lower.append(feats_lower[:, self.args.pre_frames:])
                rec_index_all_hands.append(feats_hand[:, self.args.pre_frames:])

        # Concatenate all chunks
        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        # Clamp token indices to valid range using torch.clamp
        rec_index_face = torch.clamp(rec_index_face, 0, self.lm.face_codebook_size - 1)
        rec_index_upper = torch.clamp(rec_index_upper, 0, self.lm.upper_codebook_size - 1)
        rec_index_lower = torch.clamp(rec_index_lower, 0, self.lm.lower_codebook_size - 1)
        rec_index_hands = torch.clamp(rec_index_hands, 0, self.lm.hand_codebook_size - 1)

        # Rest of the function remains unchanged - decode tokens, process poses, etc.
        rec_face = self.vae_face.decode(rec_index_face.int())
        rec_upper = self.vae_upper.decode(rec_index_upper.int())
        rec_lower = self.vae_lower.decode(rec_index_lower.int())
        rec_hands = self.vae_hand.decode(rec_index_hands.int())
        rec_face = rec_face.float()
        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rotation_6d_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper.cuda(), JOINT_MASK_UPPER, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower.cuda(), JOINT_MASK_LOWER, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands.cuda(), JOINT_MASK_HANDS, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = matrix_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        if to_global.shape[2] == 54:
            to_global = F.pad(to_global, (0, 7))
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.vae_global(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose_6d = tar_pose_6d[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]
        rec_pose_6d = axis_angle_to_6d(rec_pose.reshape(bs * n, j, 3)).reshape(bs, n, j * 6)

        tar_pose = rotation_6d_to_axis_angle(tar_pose_6d.reshape(bs*n, j, 6)).reshape(bs * n, j* 3)

        vertices_rec =self.smplx(
            betas=tar_beta.reshape(bs*n, 300),
            transl=rec_trans.reshape(bs*n, 3) - rec_trans.reshape(bs*n, 3),
            expression=tar_exps.reshape(bs*n, 100) - tar_exps.reshape(bs*n, 100),
            jaw_pose=rec_pose[:, 66:69],
            global_orient=rec_pose[:, :3],
            body_pose=rec_pose[:, 3:21 * 3 + 3],
            left_hand_pose=rec_pose[:, 25 * 3:40 * 3],
            right_hand_pose=rec_pose[:, 40 * 3:55 * 3],
            return_joints=True,
            leye_pose=rec_pose[:, 69:72],
            reye_pose=rec_pose[:, 72:75],
        )

        vertices_rec_face = self.smplx(
            betas=tar_beta.reshape(bs*n, 300),
            transl=rec_trans.reshape(bs*n, 3) - rec_trans.reshape(bs*n, 3),
            expression=rec_exps.reshape(bs*n, 100),
            jaw_pose=rec_pose[:, 66:69],
            global_orient=rec_pose[:, :3] - rec_pose[:, :3],
            body_pose=rec_pose[:, 3:21 * 3 + 3] - rec_pose[:, 3:21 * 3 + 3],
            left_hand_pose=rec_pose[:, 25 * 3:40 * 3] - rec_pose[:, 25 * 3:40 * 3],
            right_hand_pose=rec_pose[:, 40 * 3:55 * 3] - rec_pose[:, 40 * 3:55 * 3],
            return_verts=True,
            return_joints=True,
            leye_pose=rec_pose[:, 69:72] - rec_pose[:, 69:72],
            reye_pose=rec_pose[:, 72:75] - rec_pose[:, 72:75],
        )


        vertices_tar_face = self.smplx(
            betas=tar_beta.reshape(bs*n, 300),
            transl=tar_trans.reshape(bs*n, 3) - tar_trans.reshape(bs*n, 3),
            expression=tar_exps.reshape(bs*n, 100),
            jaw_pose=tar_pose[:, 66:69],
            global_orient=tar_pose[:, :3] - tar_pose[:, :3],
            body_pose=tar_pose[:, 3:21 * 3 + 3] - tar_pose[:, 3:21 * 3 + 3],
            left_hand_pose=tar_pose[:, 25 * 3:40 * 3] - tar_pose[:, 25 * 3:40 * 3],
            right_hand_pose=tar_pose[:, 40 * 3:55 * 3] - tar_pose[:, 40 * 3:55 * 3],
            return_verts=True,
            return_joints=True,
            leye_pose=tar_pose[:, 69:72] - tar_pose[:, 69:72],
            reye_pose=tar_pose[:, 72:75] - tar_pose[:, 72:75],
        )


        # Return the result set
        rs_set = {
            "rec_pose": rec_pose_6d,
            "tar_pose": tar_pose_6d,
            "tar_beta": tar_beta,
            "rec_trans": rec_trans,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "rec_exps": rec_exps,
            "vertices_rec": vertices_rec,
            "vertices_rec_face": vertices_rec_face,
            "vertices_tar_face": vertices_tar_face,
            "raw_audio": raw_audio,
        }

        return rs_set

    @torch.no_grad()
    def val_m2e_forward(self, batch):
        self.hparams.metrics_dict = []


        face_token = batch['face_token']
        hand_token = batch['hand_token']
        lower_token = batch['lower_token']
        upper_token = batch['upper_token']
        lengths = batch["m_tokens_len"]
        batch_size = face_token.shape[0]
        emotion_label = batch["emotion_label"]

        # # Motion Encode
        # motion_tokens = []
        # lengths_tokens = []
        # for i in range(len(feats_ref)):
        #     motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
        #     motion_tokens.append(motion_token[0])
        #     lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(face_tokens=face_token,
                                               hand_tokens=hand_token,
                                               upper_tokens=lower_token,
                                               lower_tokens=upper_token,
                                               emotion_label=emotion_label,
                                               lengths=lengths,
                                               task="m2e",
                                               stage='test')[-1]

        # face_tokens: Optional[Tensor] = None,
        # hand_tokens: Optional[Tensor] = None,
        # upper_tokens: Optional[Tensor] = None,
        # lower_tokens: Optional[Tensor] = None,

        # return set
        rs_set = {
            "emotion_label": emotion_label,
            "emotion_label_pred": outputs,
            # "t_ref": texts,
            # "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    @torch.no_grad()
    def val_a2t_forward(self, batch):
        self.hparams.metrics_dict = []

        audio_tokens = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        # # Motion Encode
        # motion_tokens = []
        # lengths_tokens = []
        # for i in range(len(feats_ref)):
        #     motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
        #     motion_tokens.append(motion_token[0])
        #     lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(audio_tokens=audio_tokens,
                                               lengths=lengths,
                                               task="a2t",
                                               stage='test')

        # return set
        rs_set = {
            "m_ref": audio_tokens,
            "t_ref": texts,
            "t_pred": outputs,
            "length": lengths
        }

        return rs_set

    def train_vae_forward(self, batch):

        if self.hparams.cfg.Selected_type == 'separate_rot':

            tar_pose, tar_beta, tar_trans, tar_face, tar_hand, tar_upper, tar_lower = [
                batch[key] for key in ["pose", "shape", "trans", "face", "hand", "upper", "lower"]
            ]
            lower_dim = self.hparams.cfg.Representation_type.get('separate_rot').get('lower').get('vae_test_dim')
            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            net_out_upper = self.vae_upper(tar_upper)
            tar_global = tar_lower.clone()
            tar_global[:, :, 54:57] *= 0.0
            net_out_global = self.vae_global(tar_global)
            net_out_face = self.vae_face(tar_face)
            net_out_hand = self.vae_hand(tar_hand)
            rec_lower = net_out_lower["rec_pose"]
            rec_upper = net_out_upper["rec_pose"]
            rec_global = net_out_global["rec_pose"]
            rec_face = net_out_face["rec_pose"]
            rec_hand = net_out_hand["rec_pose"]
            bs = tar_upper.shape[0]
            n = min(tar_upper.shape[1], rec_upper.shape[1])
            rec_lower = rec_lower[:,:n]
            rec_upper = rec_upper[:,:n]
            rec_global = rec_global[:,:n]
            rec_face = rec_face[:,:n]
            rec_hand = rec_hand[:,:n]
            tar_lower = tar_lower[:,:n]
            tar_upper = tar_upper[:,:n]
            tar_pose = tar_pose[:,:n]
            tar_beta = tar_beta[:,:n]
            tar_trans = tar_trans[:,:n]
            tar_face = tar_face[:,:n]
            tar_hand = tar_hand[:,:n]

            rec_pose = self.inverse_selection_tensor_full_body_6D(rec_face[:,:,:6].reshape(bs * n, 6), rec_hand.reshape(bs * n, 30 * 6), rec_lower[:,:,:54].reshape(bs * n, 9 * 6), rec_upper.reshape(bs * n, 13 * 6), JOINT_MASK_FACE_6D, JOINT_MASK_HANDS_6D, JOINT_MASK_LOWER_6D, JOINT_MASK_UPPER_6D, n*bs)
            rec_pose_aa = rotation_6d_to_axis_angle(rec_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            tar_pose_aa = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)
            rec_exps = rec_face[:, :, 6:]
            tar_exps = tar_face[:, :, 6:]
            vertices_rec = self.smplx(
                betas=tar_beta.reshape(bs*n, 300), 
                transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                expression=rec_exps.reshape(bs*n, 100), 
                jaw_pose=rec_pose_aa[:, 66:69], 
                global_orient=rec_pose_aa[:,:3], 
                body_pose=rec_pose_aa[:,3:21*3+3], 
                left_hand_pose=rec_pose_aa[:,25*3:40*3], 
                right_hand_pose=rec_pose_aa[:,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose_aa[:, 69:72], 
                reye_pose=tar_pose_aa[:, 72:75],
                )
            vertices_tar = self.smplx(
                betas=tar_beta.reshape(bs*n, 300), 
                transl=torch.zeros((bs*n, 3), device=rec_pose_aa.device), 
                expression=tar_exps.reshape(bs*n, 100), 
                jaw_pose=tar_pose_aa[:, 66:69], 
                global_orient=tar_pose_aa[:,:3], 
                body_pose=tar_pose_aa[:,3:21*3+3], 
                left_hand_pose=tar_pose_aa[:,25*3:40*3], 
                right_hand_pose=tar_pose_aa[:,40*3:55*3], 
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose_aa[:, 69:72], 
                reye_pose=tar_pose_aa[:, 72:75],
                )  


            loss_upper = self.upper_loss(rec_upper, tar_upper, tar_beta, tar_trans, self.hparams.cfg.TRAIN.Loss_6D, vertices_rec, vertices_tar)
            loss_lower = self.lower_loss(rec_lower, tar_lower, tar_beta, tar_trans, self.hparams.cfg.TRAIN.Loss_6D, vertices_rec, vertices_tar)
            loss_global = self.global_loss(rec_global, tar_lower, tar_beta, tar_trans, self.hparams.cfg.TRAIN.Loss_6D, vertices_rec, vertices_tar)

            # Extract only non-zero vectors for face and hand
            face_nonzero_mask = (tar_face.abs().sum(dim=-1).sum(dim=-1) > 0)
            hand_nonzero_mask = (tar_hand.abs().sum(dim=-1).sum(dim=-1) > 0)

            if face_nonzero_mask.any():
                loss_face = self.face_loss(rec_face[face_nonzero_mask], tar_face[face_nonzero_mask], 
                                        tar_beta[face_nonzero_mask], tar_trans[face_nonzero_mask], 
                                        self.hparams.cfg.TRAIN.Loss_6D, vertices_rec, vertices_tar)
            else:
                loss_face = torch.tensor(0.0)

            if hand_nonzero_mask.any():
                loss_hand = self.hand_loss(rec_hand[hand_nonzero_mask], tar_hand[hand_nonzero_mask],
                                        tar_beta[hand_nonzero_mask], tar_trans[hand_nonzero_mask],
                                        self.hparams.cfg.TRAIN.Loss_6D, vertices_rec, vertices_tar)
            else:
                loss_hand = torch.tensor(0.0)

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            rs_set = {
                "tar_upper":tar_upper,
                "tar_lower":tar_lower,
                "rec_upper":rec_upper,
                "rec_lower":rec_lower,
                "rec_global":rec_global,
                "rec_face":rec_face,
                "rec_hand":rec_hand,
                "tar_beta":tar_beta,
                "tar_trans":tar_trans,
                "recons-upper_loss": loss_upper,
                "commit-upper_loss": net_out_upper["embedding_loss"],
                "recons-global_loss": loss_global,
                "recons-lower_loss": loss_lower,
                "commit-lower_loss": net_out_lower["embedding_loss"],
                "recons-face_loss": loss_face,
                "commit-face_loss": net_out_face["embedding_loss"],
                "recons-hand_loss": loss_hand,
                "commit-hand_loss": net_out_hand["embedding_loss"],
                "length": lengths  # Use the computed lengths
            }

            
        elif self.hparams.cfg.Selected_type == 'full_rot':
            tar_pose = batch["pose"]
            tar_beta = batch["shape"]
            tar_trans = batch["trans"]
            tar_exps = batch["exps"]
            tar_face = batch["face"]
            tar_hand = batch["hand"]
            tar_upper = batch["upper"]
            tar_lower = batch["lower"]


        else:
            # batch detach
            feats_ref = batch["motion"]
            joints_ref = self.feats2joints(feats_ref)
            # motion encode & decode
            feats_rst, loss_commit, perplexity = self.vae(feats_ref)
            joints_rst = self.feats2joints(feats_rst)
            # return set
            rs_set = {
                "m_ref": feats_ref,
                "joints_ref": joints_ref,
                "m_rst": feats_rst,
                "joints_rst": joints_rst,
                "loss_commit": loss_commit,
                "perplexity": perplexity,
            }



        return rs_set

    @torch.no_grad()
    def val_vae_forward(self, batch, split="train"):
        self.vae_face.eval()
        self.vae_upper.eval()
        self.vae_hand.eval()
        self.vae_lower.eval()
        self.vae_global.eval()


        if self.hparams.cfg.Selected_type == 'separate_rot':

            tar_pose, tar_beta, tar_trans, tar_face, tar_hand, tar_upper, tar_lower = [
                batch[key] for key in ["pose", "shape", "trans", "face", "hand", "upper", "lower"]
            ]
            lower_dim = self.hparams.cfg.Representation_type.get('separate_rot').get('lower').get('vae_test_dim')
            net_out_lower = self.vae_lower(tar_lower[..., :lower_dim])
            net_out_upper = self.vae_upper(tar_upper)
            net_out_face = self.vae_face(tar_face)
            net_out_hand = self.vae_hand(tar_hand)

            rec_lower = net_out_lower["rec_pose"]
            rec_upper = net_out_upper["rec_pose"]
            rec_face = net_out_face["rec_pose"]
            rec_hand = net_out_hand["rec_pose"]

            bs = tar_upper.shape[0]
            n = min(tar_upper.shape[1], rec_upper.shape[1])
            rec_lower = rec_lower[:,:n]
            rec_upper = rec_upper[:,:n]
            rec_face = rec_face[:,:n]
            rec_hand = rec_hand[:,:n]
            tar_lower = tar_lower[:,:n]
            tar_upper = tar_upper[:,:n]
            tar_pose = tar_pose[:,:n]
            tar_beta = tar_beta[:,:n]
            tar_trans = tar_trans[:,:n]
            tar_face = tar_face[:,:n]
            tar_hand = tar_hand[:,:n]

            rec_exps = rec_face[:, :, 6:]
            rec_pose_jaw = rec_face[:, :, :6]
            rec_pose_legs = rec_lower[:, :, :54]
            bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
            rec_pose_upper = rotation_6d_to_axis_angle(rec_upper.reshape(bs, n, 13, 6)).reshape(bs * n, 13 * 3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, JOINT_MASK_UPPER, bs*n)

            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rotation_6d_to_matrix(rec_pose_lower)
            rec_lower2global = matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
            rec_pose_lower = matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, JOINT_MASK_LOWER, bs * n)
    
            rec_pose_hands = rotation_6d_to_axis_angle(rec_hand.reshape(bs, n, 30, 6)).reshape(bs * n, 30 * 3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, JOINT_MASK_HANDS, bs * n)

            rec_pose_jaw = rotation_6d_to_axis_angle(rec_pose_jaw.reshape(bs * n, 6)).reshape(bs * n, 1 * 3)

            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
            rec_pose[:, 66:69] = rec_pose_jaw
            tar_pose = rotation_6d_to_axis_angle(tar_pose.reshape(bs * n, 55, 6)).reshape(bs * n, 55 * 3)

            to_global = rec_lower
            to_global[:, :, :54] = rec_lower2global
            if to_global.shape[2] > 54:
                to_global[:, :, 54:57] = 0.0
            else:
                to_global = torch.cat([to_global, torch.zeros((to_global.shape[0], to_global.shape[1], 61 - to_global.shape[2]), device=to_global.device)], dim=-1)
            
            net_out_global = self.vae_global(to_global)
            rec_global = net_out_global["rec_pose"]
            rec_trans_v_s = rec_global[:, :, 54:57]
            rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)


            # Create zero expression parameters
            expression = torch.zeros((bs * n, 100), device=tar_beta.device)
            tar_beta = tar_beta.reshape(bs * n, 300)
            tar_trans = tar_trans.reshape(bs * n, 3)
            rec_trans = rec_trans.reshape(bs * n, 3)
            # Get SMPL vertices and joints for both predicted and target poses
            output_rec = self.smplx(
                betas=tar_beta,
                expression=expression,
                transl=rec_trans,  # Already correct shape [300, 3]
                global_orient=rec_pose[:,:3],
                body_pose=rec_pose[:,3:21*3+3],
                jaw_pose=rec_pose[:,66:69],
                leye_pose=rec_pose[:,69:72],
                reye_pose=rec_pose[:,72:75],
                left_hand_pose=rec_pose[:,25*3:40*3],
                right_hand_pose=rec_pose[:,40*3:55*3],
                return_verts=True,
                return_joints=True
            )

            output_tar = self.smplx(
                betas=tar_beta,
                expression=expression,
                transl=tar_trans,  # Already correct shape [300, 3]
                global_orient=tar_pose[:,:3],
                body_pose=tar_pose[:,3:21*3+3],
                jaw_pose=tar_pose[:,66:69],
                leye_pose=tar_pose[:,69:72],
                reye_pose=tar_pose[:,72:75],
                left_hand_pose=tar_pose[:,25*3:40*3],
                right_hand_pose=tar_pose[:,40*3:55*3],
                return_verts=True,
                return_joints=True
            )

            # ####  SAVE
            # # print(rec_pose.shape, tar_pose.shape)
            # tar_beta_np = tar_beta.detach().cpu().numpy()[0,0].reshape(300)
            # rec_pose_np = rec_pose.detach().cpu().numpy().reshape(bs * n, 55, 3)
            # rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
            # rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
            # gt_pose_np = tar_pose.detach().cpu().numpy().reshape(bs * n, 55, 3)
            # gt_trans_np = tar_trans.detach().cpu().numpy().reshape(bs * n, 3)
            # np.savez(pjoin('check_rec.npz'),
            #         betas=tar_beta_np,
            #         poses=rec_pose_np,
            #         expressions=rec_exp_np,
            #         trans=rec_trans_np,
            #         model='smplx2020',
            #         gender='neutral',
            #         mocap_frame_rate=30,
            #         )
            # np.savez(pjoin('check_gt.npz'),
            #         betas=tar_beta_np,
            #         poses=gt_pose_np,
            #         expressions=rec_exp_np,
            #         trans=gt_trans_np,
            #         model='smplx2020',
            #         gender='neutral',
            #         mocap_frame_rate=30,
            #         )

            # Get lengths if available, otherwise use full sequence length
            lengths = batch.get("motion_len", None)

            # rs_set = {
            #     "rec_pose":rec_pose,
            #     "tar_pose":tar_pose,
            #     "rec_trans":rec_trans,
            #     "tar_trans":tar_trans,
            #     "rec_global":rec_global,
            #     "tar_beta":tar_beta,
            #     "length": lengths  # Use the computed lengths
            # }
            rs_set = {
                "rec_joints": output_rec.joints[:, :22].reshape(bs, n, 22, 3),
                "tar_joints": output_tar.joints[:, :22].reshape(bs, n, 22, 3),
                "rec_vertices": output_rec.vertices.reshape(bs, n, 10475, 3),
                "tar_vertices": output_tar.vertices.reshape(bs, n, 10475, 3),
                "length": lengths  # Use the computed lengths
            }



        elif self.hparams.cfg.Selected_type == 'full_rot':
            tar_pose = batch["pose"]
            tar_beta = batch["shape"]
            tar_trans = batch["trans"]
            tar_exps = batch["exps"]
            tar_face = batch["face"]
            tar_hand = batch["hand"]
            tar_upper = batch["upper"]
            tar_lower = batch["lower"]

        else:
            # batch detach
            feats_ref = batch["motion"]
            joints_ref = self.feats2joints(feats_ref)
            # motion encode & decode
            feats_rst, loss_commit, perplexity = self.vae(feats_ref)
            joints_rst = self.feats2joints(feats_rst)
            # return set
            rs_set = {
                "m_ref": feats_ref,
                "joints_ref": joints_ref,
                "m_rst": feats_rst,
                "joints_rst": joints_rst,
                "loss_commit": loss_commit,
                "perplexity": perplexity,
            }


        return rs_set

    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        if (self.hparams.stage == "vae" or self.hparams.stage == "vq") and split in ["train"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

        elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_pretrain_audio"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae" or self.hparams.stage == "vq":
                rs_set = self.val_vae_forward(batch, split)

                # Update metrics
                for metric in self.hparams.metrics_dict:
                    if metric == "RotationMetrics":
                        getattr(self.metrics, metric).update(
                            rec_joints=rs_set["rec_joints"],
                            tar_joints=rs_set["tar_joints"],
                            rec_vertices=rs_set["rec_vertices"],
                            tar_vertices=rs_set["tar_vertices"],
                            lengths=rs_set['length']  # Pass the computed lengths
                        )
                    elif metric == "H3DMetrics":
                        # Convert poses to joint positions first
                        rec_joints = self.feats2joints(rs_set["rec_pose"])
                        tar_joints = self.feats2joints(rs_set["tar_pose"])
                        getattr(self.metrics, metric).update(
                            rec_pose=rec_joints,
                            tar_pose=tar_joints,
                            lengths=rs_set['length']  # Pass the computed lengths
                        )

            elif self.hparams.stage in ["lm_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["a2t"]:
                    rs_set = self.val_a2t_forward(batch)
                elif self.hparams.task in ["a2m"]:
                    rs_set = self.val_a2m_forward(batch)
                elif self.hparams.task in ["at2m"]:
                    rs_set = self.val_at2m_forward(batch)
                elif self.hparams.task in ["m2e"]:
                    rs_set = self.val_m2e_forward(batch)

            if self.hparams.task not in ["m2t", "m2e", "a2t"] and self.hparams.stage not in ["vae", "vq"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                for metric in metrics_dicts:

                    if metric == "CoSpeechMetrics":
                            motion_lengths = batch['m_tokens_len']
                            getattr(self.metrics, metric).update(
                                rec_pose=rs_set['rec_pose'],
                                tar_pose=rs_set['tar_pose'],
                                vertices_rec=rs_set['vertices_rec'],
                                vertices_rec_face=rs_set['vertices_rec_face'],
                                vertices_tar_face=rs_set['vertices_tar_face'],
                                raw_audio=rs_set['raw_audio'],
                                motion_lengths=motion_lengths,
                            )
                    elif metric == "TM2TMetrics":
                        lengths = batch['length']
                        if self.hparams.stage in [ "lm_instruct", "lm_pretrain"]:
                            word_embs = batch['word_embs']
                            pos_ohot = batch['pos_ohot']
                            text_lengths = batch['text_len']
                            if self.trainer.datamodule.is_mm:
                                word_embs = word_embs.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                pos_ohot = pos_ohot.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                                text_lengths = text_lengths.repeat_interleave(
                                    self.hparams.cfg.METRIC.MM_NUM_REPEATS,
                                    dim=0)
                        else:
                            word_embs = None
                            pos_ohot = None
                            text_lengths = None

                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            feats_rst=rs_set["m_rst"],
                            lengths_ref=lengths,
                            lengths_rst=rs_set['length'],
                            word_embs=word_embs,
                            pos_ohot=pos_ohot,
                            text_lengths=text_lengths,
                        )
                    elif metric == "MMMetrics":
                        # pass
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain" ]:
                # self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
                for metric in metrics_dicts:
                    if metric == "M2TMetrics":
                        getattr(self.metrics, metric).update(
                            feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set["t_pred"],
                            gt_texts=batch["all_captions"],
                            lengths=rs_set['length'],
                            word_embs=batch["word_embs"],
                            pos_ohot=batch["pos_ohot"],
                            text_lengths=batch["text_len"],
                        )
            elif self.hparams.task == "m2e" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_rl"
            ]:
                # self.hparams.metrics_dict = metrics_dicts = ['M2EMetrics']
                for metric in metrics_dicts:
                    if metric == "M2EMetrics":
                        # for index_pre in range(len(rs_set["emotion_label_pred"])):
                        #     rs_set["emotion_label_pred"][index_pre] = np.random.choice(["sadness", "contempt", "neutral", "fear", "anger", "happiness", "disgust", "surprise"])
                        # for index_pre in range(len(rs_set["emotion_label_pred"])):
                        #     rs_set["emotion_label_pred"][index_pre] = "contempt"

                        # for index_pre in range(len(rs_set["emotion_label_pred"])):
                        #     rs_set["emotion_label_pred"][index_pre] = batch["emotion_label"][index_pre]

                        getattr(self.metrics, metric).update(
                            # feats_ref=rs_set["m_ref"],
                            pred_texts=rs_set['emotion_label_pred'],
                            gt_texts=batch["emotion_label"],
                            lengths=batch['m_tokens_len'],
                            word_embs=batch['word_embeddings'],
                            pos_ohot=batch['pos_one_hots'],
                            text_lengths=batch['sent_len'],
                        )
            elif self.hparams.task == "a2t" and self.hparams.stage in [
                    "lm_pretrain_audio",
            ]:
                # self.hparams.metrics_dict = metrics_dicts = ['A2TMetrics']
                for metric in metrics_dicts:
                    if metric == "A2TMetrics":
                        getattr(self.metrics, metric).update(
                            pred_texts=rs_set["t_pred"],
                            gt_texts=rs_set["t_ref"],
                        )
        # return forward output rather than loss during test

        if split in ["test"] and self.hparams.stage != "vq":
            if self.hparams.task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set["joints_ref"]
                # pass
            elif self.hparams.task == "m2e":
                return rs_set["emotion_label_pred"], batch["m_tokens_len"]
            elif self.hparams.task == "a2m":
                return rs_set["rec_pose"], rs_set["tar_pose"], rs_set["tar_beta"], rs_set["rec_trans"], rs_set["tar_trans"], rs_set["tar_exps"], rs_set["rec_exps"]

        return loss
