import numpy as np
import os
import random
import torch
import time
from mGPT.config import instantiate_from_config
from os.path import join as pjoin
from mGPT.losses.mgpt import GPTLosses
from mGPT.models.base import BaseModel
from .base import BaseModel
import json
import mGPT.render.matplot.plot_3d_global as plot_3d
import jiwer
from mGPT.archs.motion_representation import VQVAEConvZero, VAEConvZero
from utils_emage import rotation_conversions as rc
from utils_emage import other_tools
from mGPT.data.beat2.data_tools import joints_list
import torch.nn.functional as F
vq_model_module = __import__(f"mGPT.archs.motion_representation", fromlist=["something"])


class ExpressionalMotionGPT(BaseModel):
    """
    Stage 1 Motion Tokenizer
    Stage 2 Motion-language pretrian
    Stage 3 Motion-language instruction tuning
    """

    def __init__(self,
                 cfg,
                 datamodule,
                 lm,
                 motion_vae,
                 audio_setup,
                 codebook_size=512,
                 stage='vae',
                 debug=True,
                 condition='text',
                 task='t2m',
                 metrics_dict=['TM2TMetrics'],
                 **kwargs):

        self.save_hyperparameters(ignore='datamodule', logger=False)
        self.datamodule = datamodule
        super().__init__()

        self.args = cfg.DATASET
        self.vary_length = cfg.DATASET.vary_length

        self.audio_samplerate = audio_setup['params']['audio_samplerate']
        self.audio_down = audio_setup['params']['audio_down']
        self.motion_fps = cfg.DATASET.pose_fps

        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
        ori_joints = 'beat_smplx_joints'
        tar_joints = 'beat_smplx_full'
        self.ori_joint_list = joints_list[ori_joints]
        self.tar_joint_list = joints_list[tar_joints]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[
            self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[
            self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[
            self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[
            self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        # Instantiate motion tokenizer
        if motion_vae != None:
            cfg.vq.emage.params.vae_layer = 3
            cfg.vq.emage.params.vae_length = 256
            # create face model
            cfg.vq.emage.params.vae_test_dim = 106
            # self.vq_model_face = VQVAEConvZero(cfg.vq.emage.params)
            self.vq_model_face = getattr(vq_model_module, cfg.TRAIN.VAE_NAME)(cfg.vq.emage.params)

            # create upper body model
            cfg.vq.emage.params.vae_test_dim = 78
            # self.vq_model_upper = VQVAEConvZero(cfg.vq.emage.params)
            self.vq_model_upper = getattr(vq_model_module, cfg.TRAIN.VAE_NAME)(cfg.vq.emage.params)

            # create hands model
            cfg.vq.emage.params.vae_test_dim = 180
            # self.vq_model_hands = VQVAEConvZero(cfg.vq.emage.params)
            self.vq_model_hands = getattr(vq_model_module, cfg.TRAIN.VAE_NAME)(cfg.vq.emage.params)

            # create lower model
            # cfg.vq.emage.params.vae_test_dim = 61
            # cfg.vq.emage.params.vae_layer = 4

            cfg.vq.emage.params.vae_test_dim = 54
            cfg.vq.emage.params.vae_layer = 3
            # self.vq_model_lower = VQVAEConvZero(cfg.vq.emage.params)
            self.vq_model_lower = getattr(vq_model_module, cfg.TRAIN.VAE_NAME)(cfg.vq.emage.params)


            # create foot model
            cfg.vq.emage.params.vae_test_dim = 61
            cfg.vq.emage.params.vae_layer = 4
            self.global_motion = VAEConvZero(cfg.vq.emage.params)

        # Instantiate motion-language model
        self.lm = instantiate_from_config(lm)

        # Freeze the motion tokenizer for lm training
        if 'lm' in self.hparams.stage:
            # self.vae.training = False
            self.vq_model_face.training = False
            for p in self.vq_model_face.parameters():
                p.requires_grad = False
            self.vq_model_upper.training = False
            for p in self.vq_model_upper.parameters():
                p.requires_grad = False
            self.vq_model_hands.training = False
            for p in self.vq_model_hands.parameters():
                p.requires_grad = False
            self.vq_model_lower.training = False
            for p in self.vq_model_lower.parameters():
                p.requires_grad = False
            self.global_motion.training = False
            for p in self.global_motion.parameters():
                p.requires_grad = False


        # Instantiate the losses
        self._losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, self.hparams.stage, self.datamodule.njoints)
            for split in ["losses_train", "losses_test", "losses_val"]
        })

        # Data transform
        self.feats2joints = datamodule.feats2joints

        # Count codebook frequency
        self.codePred = []
        self.codeFrequency = torch.zeros((self.hparams.codebook_size, ))

    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t



    def forward(self, batch, task="t2m"):
        # texts = batch["text"]
        # lengths_ref = batch["length"]
        if "audio" in batch:
            audio_token_string = batch["audio"]
            audio_token_string = ['Generate motion: ' + audio for audio in audio_token_string]
            outputs_face, outputs_hand, outputs_lower, outputs_upper, output_texts = self.lm.generate_direct(audio_tokens=audio_token_string, do_sample=True)

            feats_face = torch.zeros([len(outputs_face), 64]).to(self.device)
            feats_hand = torch.zeros([len(outputs_hand), 64]).to(self.device)
            feats_upper = torch.zeros([len(outputs_upper), 64]).to(self.device)
            feats_lower = torch.zeros([len(outputs_lower), 64]).to(self.device)

            # padding and concat
            for i in range(len(feats_face)):
                if outputs_face[i].shape[0] <= 64:
                    feats_face[i, :outputs_face[i].shape[0]] = outputs_face[i]
                else:
                    feats_face[i, :64] = outputs_face[i][:64]
            # padding and concat
            for i in range(len(feats_hand)):
                if outputs_hand[i].shape[0] <= 64:
                    feats_hand[i, :outputs_hand[i].shape[0]] = outputs_hand[i]
                else:
                    feats_hand[i, :64] = outputs_hand[i][:64]
            # padding and concat
            for i in range(len(feats_upper)):
                if outputs_upper[i].shape[0] <= 64:
                    feats_upper[i, :outputs_upper[i].shape[0]] = outputs_upper[i]
                else:
                    feats_upper[i, :64] = outputs_upper[i][:64]
            # padding and concat
            for i in range(len(feats_lower)):
                if outputs_lower[i].shape[0] <= 64:
                    feats_lower[i, :outputs_lower[i].shape[0]] = outputs_lower[i]
                else:
                    feats_lower[i, :64] = outputs_lower[i][:64]




        if "text" in batch:

            rec_index_all_face = []
            rec_index_all_upper = []
            rec_index_all_lower = []
            rec_index_all_hands = []
            text_token_string = batch["text"]
            # text_token_string = ['Generate motion from given caption: ' + text for text in text_token_string]
            outputs_face, outputs_hand, outputs_lower, outputs_upper, output_texts = self.lm.generate_direct(input=text_token_string, do_sample=True)

            output_length = max(len(outputs_face[0]), len(outputs_hand[0]), len(outputs_lower[0]), len(outputs_upper[0]))
            feats_face, feats_hand, feats_lower, feats_upper = self.unify_length(outputs_face, outputs_hand,outputs_lower, outputs_upper, output_length)

            feats_face = torch.stack(feats_face, dim=0)
            feats_hand = torch.stack(feats_hand, dim=0)
            feats_lower = torch.stack(feats_lower, dim=0)
            feats_upper = torch.stack(feats_upper, dim=0)

            rec_index_all_face.append(feats_face)
            rec_index_all_upper.append(feats_upper)
            rec_index_all_lower.append(feats_lower)
            rec_index_all_hands.append(feats_hand)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_face = self.vq_model_face.decode(rec_index_face.int())
        rec_upper = self.vq_model_upper.decode(rec_index_upper.int())
        rec_lower = self.vq_model_lower.decode(rec_index_lower.int())
        rec_hands = self.vq_model_hands.decode(rec_index_hands.int())

        j = 55
        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        if to_global.shape[2] == 54:
            to_global = F.pad(to_global, (0, 7))

        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, torch.zeros_like(rec_trans_v_s[:, 0, 0:1])  )
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, torch.zeros_like(rec_trans_v_s[:, 0, 2:3]))
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)

        # rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        # rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        #
        # ####  SAVE
        # # print(rec_pose.shape, tar_pose.shape)
        # rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
        # rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
        rec_trans = rec_trans.reshape(bs * n, 3)
        rec_exps = rec_exps.reshape(bs * n, 100)

        if output_length != n:
            output_length = n

        # return set
        outputs = {
            "texts": output_texts,
            "rec_pose": rec_pose,
            "rec_trans":rec_trans,
            "rec_exps":rec_exps,
            "length": output_length
        }

        return outputs

    def train_lm_forward(self, batch):


        face_token = batch['face_token']
        hand_token = batch['hand_token']
        lower_token = batch['lower_token']
        upper_token = batch['upper_token']
        # m_tokens_len = batch['m_tokens_len']
        lengths = batch["m_tokens_len"]

        batch_size = face_token.shape[0]

        # texts = batch["text"]
        if "text" in batch:
            texts = batch["text"]
            text_timestamp = batch["text_timestamp"]

        else:
            texts = [None] * batch_size
            text_timestamp = [None] * batch_size

        if 'audio_token' in batch:
            audio_tokens = batch['audio_token']
            audio_length = batch['a_tokens_len']
        else:
            audio_tokens = [None] * batch_size
            audio_length = [None] * batch_size
        tasks = batch["tasks"]


        # LLM Forward
        outputs = self.lm(texts, text_timestamp,face_token, hand_token, lower_token, upper_token, audio_tokens, lengths, audio_length, tasks)

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

        min_len = lengths.copy()
        # Forward
        outputs = self.lm.generate_conditional(texts,
                                               lengths=lengths,
                                               stage='test',
                                               tasks=tasks)

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(texts)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

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

    def pad_tensor(self, tensor, target_length):
        """Pad the given tensor to the target length or truncate it to the target length."""
        current_length = tensor.size(0)

        if current_length < target_length:
            # 使用0填充到目标长度，并确保填充的 Tensor 在与原始 Tensor 相同的设备上
            padding_length = target_length - current_length
            return torch.cat([tensor, torch.zeros(padding_length, device=tensor.device)], dim=0)
        elif current_length > target_length:
            # 如果长度超过目标长度，则进行截断
            return tensor[:target_length]
        return tensor

    def unify_length(self, outputs_face, outputs_hand, outputs_lower, outputs_upper, motion_length):
        """Unify the length of all tensors in all lists to the maximum length among the four lists."""

        # max_length = max(
        #     max(tensor.size(0) for tensor in outputs_face),
        #     max(tensor.size(0) for tensor in outputs_hand),
        #     max(tensor.size(0) for tensor in outputs_lower),
        #     max(tensor.size(0) for tensor in outputs_upper)
        # )
        max_length = motion_length

        # 统一四个列表中每个 tensor 的长度
        outputs_face = [self.pad_tensor(tensor, max_length) for tensor in outputs_face]
        outputs_hand = [self.pad_tensor(tensor, max_length) for tensor in outputs_hand]
        outputs_lower = [self.pad_tensor(tensor, max_length) for tensor in outputs_lower]
        outputs_upper = [self.pad_tensor(tensor, max_length) for tensor in outputs_upper]

        return outputs_face, outputs_hand, outputs_lower, outputs_upper

    @torch.no_grad()
    def val_a2m_forward(self, batch):
        # feats_ref = batch["motion"]
        face = batch['face']
        hand = batch['hand']
        lower = batch['lower']
        upper = batch['upper']
        tar_pose = batch["tar_pose"]
        tar_beta = batch["tar_beta"]
        tar_trans = batch["tar_trans"]
        tar_exps = batch["tar_exps"]
        lengths = batch["m_tokens_len"]
        batch_size = face.shape[0]

        # texts = batch["text"]
        if "text" in batch:
            texts = batch["text"]
        else:
            texts = [None] * batch_size
        if 'audio_token' in batch:
            audio_tokens = batch['audio_token']
            raw_audio = batch['raw_audio']
            audio_length = batch['a_tokens_len']
        else:
            audio_tokens = [None] * batch_size
            audio_length = [None] * batch_size
        # tasks = batch["tasks"]


        bs, n = tar_pose.shape[0], tar_pose.shape[1]

        # ### DEBUG ###
        # n = 128

        j = 55
        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            # tar_exps = tar_exps[:, :-remain, :]
            # tar_contact = tar_contact[:, :-remain, :]
            face = face[:, :-remain, :]
            hand = hand[:, :-remain, :]
            lower =lower[:, :-remain, :]
            upper = upper[:, :-remain, :]
            n = n - remain


        if self.vary_length == False:


            pass

        else:

            roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
            remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
            round_l = self.args.pose_length - self.args.pre_frames

            rec_index_all_face = []
            rec_index_all_upper = []
            rec_index_all_lower = []
            rec_index_all_hands = []

            texts = [None] * bs
            tasks = None
            for i in range(0, roundt):
                # audio fps is 16000/320 and pose fps is 30
                audio_tokens_tmp = audio_tokens[:, i*int(self.audio_samplerate/self.audio_down/self.motion_fps*round_l): (i+1)*int(self.audio_samplerate/self.audio_down/self.motion_fps*round_l) + int(self.audio_samplerate/self.audio_down/self.motion_fps*self.args.pre_frames)]
                motion_lengths = [self.args.pose_length] * bs

                # Forward
                outputs_face, outputs_hand, outputs_lower, outputs_upper = self.lm.generate_conditional(texts=texts,
                                                                               audio_tokens=audio_tokens_tmp,
                                                                               lengths=lengths,
                                                                               audio_lengths=[len(audio_token) for audio_token in audio_tokens_tmp],
                                                                               task='a2m',
                                                                               stage='test',
                                                                               tasks=tasks)

                feats_face, feats_hand, feats_lower, feats_upper = self.unify_length(outputs_face, outputs_hand, outputs_lower, outputs_upper, 64)

                feats_face = torch.stack(feats_face, dim=0)
                feats_hand = torch.stack(feats_hand, dim=0)
                feats_lower = torch.stack(feats_lower, dim=0)
                feats_upper = torch.stack(feats_upper, dim=0)

                # feats_face = torch.zeros([bs, 64]).to(self.device)
                # feats_hand = torch.zeros([bs, 64]).to(self.device)
                # feats_lower = torch.zeros([bs, 64]).to(self.device)
                # feats_upper = torch.zeros([bs, 64]).to(self.device)
                # # padding and concat
                # for idx in range(len(feats_face)):
                #     if outputs_face[idx].shape[0] <= 64:
                #         feats_face[idx, :outputs_face[idx].shape[0]] = outputs_face[idx]
                #     else:
                #         feats_face[idx, :64] = outputs_face[idx][:64]
                # # padding and concat
                # for idx in range(len(feats_hand)):
                #     if outputs_hand[idx].shape[0] <= 64:
                #         feats_hand[idx, :outputs_hand[idx].shape[0]] = outputs_hand[idx]
                #     else:
                #         feats_hand[idx, :64] = outputs_hand[idx][:64]
                # # padding and concat
                # for idx in range(len(feats_upper)):
                #     if outputs_upper[idx].shape[0] <= 64:
                #         feats_upper[idx, :outputs_upper[idx].shape[0]] = outputs_upper[idx]
                #     else:
                #         feats_upper[idx, :64] = outputs_upper[idx][:64]
                # # padding and concat
                # for idx in range(len(feats_lower)):
                #     if outputs_lower[idx].shape[0] <= 64:
                #         feats_lower[idx, :outputs_lower[idx].shape[0]] = outputs_lower[idx]
                #     else:
                #         feats_lower[idx, :64] = outputs_lower[idx][:64]

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


        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_face = self.vq_model_face.decode(rec_index_face.int())
        rec_upper = self.vq_model_upper.decode(rec_index_upper.int())
        rec_lower = self.vq_model_lower.decode(rec_index_lower.int())
        rec_hands = self.vq_model_hands.decode(rec_index_hands.int())

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)  #
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs * n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs * n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs * n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs * n, 1 * 3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]
        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)

        ####  SAVE
        # print(rec_pose.shape, tar_pose.shape)
        # rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j, 6))
        # rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs * n, j * 3)
        # tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        # tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

        # return set
        rs_set = {
            "rec_pose": rec_pose,
            "tar_pose": tar_pose,
            "tar_beta": tar_beta,
            "rec_trans": rec_trans,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "rec_exps": rec_exps,
            "raw_audio":raw_audio,
        }

        return rs_set



    @torch.no_grad()
    def val_m2t_forward(self, batch):
        self.hparams.metrics_dict = []

        feats_ref = batch["motion"]
        texts = batch["text"]
        lengths = batch["length"]
        all_captions = batch['all_captions']

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])
            lengths_tokens.append(motion_token.shape[1])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths_tokens,
                                               task="m2t",
                                               stage='test')

        # return set
        rs_set = {
            "m_ref": feats_ref,
            "t_ref": all_captions,
            # "t_ref": texts,
            "t_pred": outputs,
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

    @torch.no_grad()
    def val_m2m_forward(self, batch, task="pred"):
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Motion Encode
        motion_tokens = []
        lengths_tokens = []
        for i in range(len(feats_ref)):
            motion_token, _ = self.vae.encode(feats_ref[i:i + 1])
            motion_tokens.append(motion_token[0])

        # Forward
        outputs = self.lm.generate_conditional(motion_tokens=motion_tokens,
                                               lengths=lengths,
                                               task=task,
                                               stage='test')

        # Motion Decode
        feats_rst = torch.zeros_like(feats_ref)
        min_len = lengths.copy()

        for i in range(len(lengths)):
            outputs[i] = torch.clamp(outputs[i],
                                     0,
                                     self.hparams.codebook_size - 1,
                                     out=None)

            if len(outputs[i]) > 1:
                motion = self.vae.decode(outputs[i])
            else:
                motion = torch.zeros_like(feats_ref[i:i + 1, ...])

            min_len[i] = min(motion.shape[1], lengths[i])

            # Cut Motion
            feats_rst[i:i + 1, :min_len[i], ...] = motion[:, :lengths[i]]

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

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

    def train_vae_forward(self, batch):
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
        # Detach batch
        feats_ref = batch["motion"]
        lengths = batch["length"]

        # Repeat for multimodal evaluation
        if self.trainer.datamodule.is_mm:
            feats_ref = feats_ref.repeat_interleave(
                self.hparams.cfg.METRIC.MM_NUM_REPEATS, dim=0)
            lengths = lengths * self.hparams.cfg.METRIC.MM_NUM_REPEATS

        # Motion encode & decode
        feats_rst = torch.zeros_like(feats_ref)

        for i in range(len(feats_ref)):
            if lengths[i] == 0:
                continue
            feats_pred, _, _ = self.vae(feats_ref[i:i + 1, :lengths[i]])
            feats_rst[i:i + 1, :feats_pred.shape[1], :] = feats_pred

            code_pred, _ = self.vae.encode(feats_ref[i:i + 1, :lengths[i]])

            # codeFre_pred = torch.bincount(code_pred[0],
            #                               minlength=self.hparams.codebook_size).to(
            #                                   self.codeFrequency.device)
            # self.codePred.append(code_pred[0])
            # self.codeFrequency += codeFre_pred

        # np.save('../memData/results/codeFrequency.npy',
        #         self.codeFrequency.cpu().numpy())

        # Recover joints for evaluation
        joints_ref = self.feats2joints(feats_ref)
        joints_rst = self.feats2joints(feats_rst)

        # Renorm for evaluation
        feats_ref = self.datamodule.renorm4t2m(feats_ref)
        feats_rst = self.datamodule.renorm4t2m(feats_rst)

        # Return set
        rs_set = {
            "m_ref": feats_ref,
            "joints_ref": joints_ref,
            "m_rst": feats_rst,
            "joints_rst": joints_rst,
            "length": lengths,
        }

        return rs_set


    def allsplit_step(self, split: str, batch, batch_idx):
        # Compute the losses
        loss = None

        if self.hparams.stage == "vae" and split in ["train", "val"]:
            rs_set = self.train_vae_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage in ["lm_instruct", "lm_causal_instruct", "lm_pretrain"] and split in ["train"]:
            rs_set = self.train_lm_forward(batch)
            loss = self._losses['losses_' + split].update(rs_set)
        elif self.hparams.stage == 'lm_rl' and split in ['train']:
            rs_set = self.train_rl_forward(batch)
            loss = None

        # Compute the metrics
        if split in ["val", "test"]:
            if self.hparams.stage == "vae":
                rs_set = self.val_vae_forward(batch, split)
            elif self.hparams.stage in ["lm_instruct", "lm_causal_instruct", "lm_pretrain", "lm_rl"]:
                if self.hparams.task == "t2m":
                    rs_set = self.val_t2m_forward(batch)
                elif self.hparams.task == "m2t":
                    rs_set = self.val_m2t_forward(batch)
                elif self.hparams.task in ["m2m", "pred", "inbetween"]:
                    rs_set = self.val_m2m_forward(batch, self.hparams.task)
                elif self.hparams.task in ["a2t"]:  ############  zhangjz #########
                    rs_set = self.val_a2t_forward(batch)
                elif self.hparams.task in ["a2m"]:  ############  zhangjz #########
                    rs_set = self.val_a2m_forward(batch)

            if self.hparams.task not in ["m2t", "a2t"]:
                # MultiModality evaluation sperately
                if self.trainer.datamodule.is_mm:
                    metrics_dicts = ['MMMetrics']
                else:
                    metrics_dicts = self.hparams.metrics_dict
                    
                if self.hparams.task not in ['pred', 'inbetween'] and 'PredMetrics' in metrics_dicts:
                    metrics_dicts.remove('PredMetrics')

                for metric in metrics_dicts:

                    if metric == "AM2AMetrics_Exp":
                            # word_embs = None
                            # pos_ohot = None
                            motion_lengths = batch['m_tokens_len']
                            getattr(self.metrics, metric).update(
                                rec_pose=rs_set['rec_pose'],
                                tar_pose=rs_set['tar_pose'],
                                tar_beta=rs_set['tar_beta'],
                                rec_trans=rs_set['rec_trans'],
                                tar_trans = rs_set['tar_trans'],
                                tar_exps=rs_set['tar_exps'],
                                rec_exps=rs_set['rec_exps'],
                                raw_audio=rs_set['raw_audio'],
                                motion_lengths=motion_lengths
                            )

                    elif metric == "UncondMetrics":
                        getattr(self.metrics, metric).update(
                            recmotion_embeddings=rs_set["lat_rm"],
                            gtmotion_embeddings=rs_set["lat_m"],
                            lengths=lengths,
                        )
                    elif metric == "MRMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "PredMetrics":
                        getattr(self.metrics,
                                metric).update(rs_set["joints_rst"],
                                               rs_set["joints_ref"], lengths)
                    elif metric == "MMMetrics":
                        # pass
                        getattr(self.metrics,
                                metric).update(rs_set["m_rst"],
                                               rs_set['length'])
                    else:
                        raise TypeError(f"Not support this metric {metric}")

            elif self.hparams.task == "m2t" and self.hparams.stage in [
                    "lm_instruct", "lm_pretrain", "lm_causal_instruct"
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['M2TMetrics']
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
            elif self.hparams.task == "a2t" and self.hparams.stage in [
                    "lm_pretrain_audio",
            ]:
                self.hparams.metrics_dict = metrics_dicts = ['A2TMetrics']
                for metric in metrics_dicts:
                    if metric == "A2TMetrics":
                        getattr(self.metrics, metric).update(
                            pred_texts=rs_set["t_pred"],
                            gt_texts=rs_set["t_ref"],
                        )
        # return forward output rather than loss during test
        if split in ["test"]:
            if self.hparams.task == "t2m":
                return rs_set["joints_rst"], rs_set["length"], rs_set["joints_ref"]
                # pass
            elif self.hparams.task == "m2t":
                return rs_set["t_pred"], batch["length"]
            elif self.hparams.task == "a2m":
                return rs_set["rec_pose"], rs_set["tar_pose"], rs_set["tar_beta"], rs_set["rec_trans"], rs_set["tar_trans"], rs_set["tar_exps"], rs_set["rec_exps"]


        return loss
