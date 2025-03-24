from typing import List
import os
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
from lom.config import instantiate_from_config
from lom.archs.motion_representation import VQVAEConvZero, VAEConvZero, VAESKConv
from utils_emage import other_tools, metric
from utils_emage import rotation_conversions as rc
from lom.optimizers.loss_factory import get_loss_func
import librosa
import smplx
from lom.metrics import data_tools
import torch.nn.functional as F
from os.path import join as pjoin

class AM2AMetrics_Emage(Metric):
    def __init__(self,
                 cfg,
                 dataname='beat2',
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.dataname = dataname
        self.name = "matching, fid, and diversity scores"
        self.text = 'lm' in cfg.TRAIN.STAGE and cfg.model.params.task == 'a2m'

        self.add_state("l2_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("l2_loss_all", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lvel_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lvel_loss_all", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fid_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("fid_score_all", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("align_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("align_score_all", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("l1div_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("l1div_score_all", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rec_pose", default=[], dist_reduce_fx="cat")
        self.add_state("tar_pose", default=[], dist_reduce_fx="cat")
        self.add_state("motion_lengths", default=[], dist_reduce_fx="cat")

        # self.add_state("latent_out", default=torch.empty(0, 240), dist_reduce_fx="cat")
        # self.add_state("latent_ori", default=torch.empty(0, 240), dist_reduce_fx="cat")
        # self.add_state("motion_length", default=torch.empty(0, 1), dist_reduce_fx="cat")

        # self.latent_out = []
        # self.latent_ori = []


        self.metrics = ["fid_score", "l2_loss", "lvel_loss"]


        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")

        self.eval_model = VAESKConv(cfg.vq.emage.params)
        other_tools.load_checkpoints(self.eval_model, pjoin(cfg.DATASET.BEAT2.ROOT, cfg.DATASET.e_path ), cfg.DATASET.e_name)
        self.eval_model.eval()

        # select trainable joints
        self.smplx = smplx.create(
            cfg.DATASET.smpl_path,
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).eval()

        self.avg_vel = np.load( pjoin( cfg.DATASET.BEAT2.ROOT, f"weights/mean_vel_{cfg.DATASET.pose_rep}.npy" ))
        self.alignmenter = metric.alignment(0.3, 7, self.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21])
        self.align_mask = 60
        self.joints = 55
        self.l1_calculator = metric.L1div()

        self.cls_loss = nn.NLLLoss()
        self.reclatent_loss = nn.MSELoss()
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.rec_loss = get_loss_func("GeodesicLoss")

    @torch.no_grad()
    def reset(self):
        # 重置 list 变量
        if isinstance(self.rec_pose, list):
            self.rec_pose.clear()
            self.tar_pose.clear()
            self.motion_lengths.clear()
        else:
            self.rec_pose = []
            self.tar_pose = []
            self.motion_lengths = []

        # 重置其他 metric
        self.l2_loss = torch.tensor(0.0).to(self.device)
        self.lvel_loss = torch.tensor(0.0).to(self.device)
        self.fid_score = torch.tensor(0.0).to(self.device)
        self.align_score = torch.tensor(0.0).to(self.device)

        self.l2_loss_all = torch.tensor(0.0).to(self.device)
        self.lvel_loss_all = torch.tensor(0.0).to(self.device)
        self.fid_score_all = torch.tensor(0.0).to(self.device)
        self.align_score_all = torch.tensor(0.0).to(self.device)

        # Reset count and length trackers
        self.count = torch.tensor(0).to(self.device)
        self.total_length = torch.tensor(0).to(self.device)

    @torch.no_grad()
    def compute(self, sanity_flag):
        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        if isinstance(self.rec_pose, list):
            self.rec_pose = torch.cat(self.rec_pose, dim=0)
            self.tar_pose = torch.cat(self.tar_pose, dim=0)
            self.motion_lengths = torch.cat(self.motion_lengths, dim=0)
        bs, n = self.rec_pose.shape[0], self.rec_pose.shape[1]
        self.latent_out = []
        self.latent_ori = []
        for batch_index in range(bs):
            motion_length = self.motion_lengths[batch_index]
            if motion_length > n:
                motion_length = n
            remain = motion_length % self.cfg.vq.emage.params.vae_test_len
            self.latent_out.append(self.eval_model.map2latent(self.rec_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())  # downsampling x16
            self.latent_ori.append(self.eval_model.map2latent(self.tar_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())

        device = self.l2_loss.device
        latent_out_all = np.concatenate(self.latent_out, axis=0)
        latent_ori_all = np.concatenate(self.latent_ori, axis=0)
        # self.fid_score = torch.tensor(data_tools.FIDCalculator.frechet_distance(self.latent_out.detach().cpu().numpy(), self.latent_ori.detach().cpu().numpy())).to(device)
        self.fid_score = torch.tensor(data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)).to(device)

        self.l2_loss =self.l2_loss_all / self.total_length
        self.lvel_loss =self.lvel_loss_all / self.total_length
        self.align_score = self.align_score_all/(self.total_length-2* self.count*self.align_mask)

        metrics["fid_score"] = self.fid_score
        metrics["l2_loss"] = self.l2_loss
        metrics["lvel_loss"] = self.lvel_loss
        metrics["align_score"] = self.align_score

        # Reset
        self.reset()

        return {**metrics}


    @torch.no_grad()
    def update(self,
        rec_pose: Tensor = None,
        tar_pose: Tensor = None,
        tar_beta: Tensor = None,
        rec_trans : Tensor = None,
        tar_trans: Tensor = None,
        tar_exps : Tensor = None,
        rec_exps: Tensor = None,
        raw_audio:Tensor = None,
        motion_lengths:Tensor = None):

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        # self.motion_lengths.append(torch.tensor(motion_lengths).to(device))
        motion_lengths[motion_lengths > n] = n
        self.motion_lengths.append(motion_lengths)
        # remain = n % self.cfg.vq.emage.params.vae_test_len

        # Calculate the amount of padding needed (for the second dimension)
        pad_amount = 3000 - rec_pose.size(1)
        rec_pose_padded = F.pad(rec_pose, (0, 0, 0, pad_amount))
        tar_pose_padded = F.pad(tar_pose, (0, 0, 0, pad_amount))

        self.rec_pose.append(rec_pose_padded)  # downsampling x16
        self.tar_pose.append(tar_pose_padded)

        for batch_index in range(bs):
            motion_length = motion_lengths[batch_index]
            # if motion_length > n:
            #     motion_length = n
            # remain = motion_length % self.cfg.vq.emage.params.vae_test_len

            # self.latent_out.append(self.eval_model.map2latent(rec_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())  # downsampling x16
            # self.latent_ori.append(self.eval_model.map2latent(tar_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())


            rec_pose_sample = rc.rotation_6d_to_matrix(rec_pose[batch_index].reshape(n, j, 6))
            rec_pose_sample = rc.matrix_to_axis_angle(rec_pose_sample).reshape(n, j * 3)
            tar_pose_sample = rc.rotation_6d_to_matrix(tar_pose[batch_index].reshape(n, j, 6))
            tar_pose_sample = rc.matrix_to_axis_angle(tar_pose_sample).reshape(n, j * 3)

            vertices_rec = self.smplx(
                betas=tar_beta[batch_index].reshape(n, 300),
                transl=rec_trans[batch_index].reshape(n, 3) - rec_trans[batch_index].reshape(n, 3),
                expression=tar_exps[batch_index].reshape(n, 100) - tar_exps[batch_index].reshape(n, 100),
                jaw_pose=rec_pose_sample[:, 66:69],
                global_orient=rec_pose_sample[:, :3],
                body_pose=rec_pose_sample[:, 3:21 * 3 + 3],
                left_hand_pose=rec_pose_sample[:, 25 * 3:40 * 3],
                right_hand_pose=rec_pose_sample[:, 40 * 3:55 * 3],
                return_joints=True,
                leye_pose=rec_pose_sample[:, 69:72],
                reye_pose=rec_pose_sample[:, 72:75],
            )

            vertices_rec_face = self.smplx(
                betas=tar_beta[batch_index].reshape(n, 300),
                transl=rec_trans[batch_index].reshape(n, 3) - rec_trans[batch_index].reshape(n, 3),
                expression=rec_exps[batch_index].reshape(n, 100),
                jaw_pose=rec_pose_sample[:, 66:69],
                global_orient=rec_pose_sample[:, :3] - rec_pose_sample[:, :3],
                body_pose=rec_pose_sample[:, 3:21 * 3 + 3] - rec_pose_sample[:, 3:21 * 3 + 3],
                left_hand_pose=rec_pose_sample[:, 25 * 3:40 * 3] - rec_pose_sample[:, 25 * 3:40 * 3],
                right_hand_pose=rec_pose_sample[:, 40 * 3:55 * 3] - rec_pose_sample[:, 40 * 3:55 * 3],
                return_verts=True,
                return_joints=True,
                leye_pose=rec_pose_sample[:, 69:72] - rec_pose_sample[:, 69:72],
                reye_pose=rec_pose_sample[:, 72:75] - rec_pose_sample[:, 72:75],
            )
            vertices_tar_face = self.smplx(
                betas=tar_beta[batch_index].reshape(n, 300),
                transl=tar_trans[batch_index].reshape(n, 3) - tar_trans[batch_index].reshape(n, 3),
                expression=tar_exps[batch_index].reshape(n, 100),
                jaw_pose=tar_pose_sample[:, 66:69],
                global_orient=tar_pose_sample[:, :3] - tar_pose_sample[:, :3],
                body_pose=tar_pose_sample[:, 3:21 * 3 + 3] - tar_pose_sample[:, 3:21 * 3 + 3],
                left_hand_pose=tar_pose_sample[:, 25 * 3:40 * 3] - tar_pose_sample[:, 25 * 3:40 * 3],
                right_hand_pose=tar_pose_sample[:, 40 * 3:55 * 3] - tar_pose_sample[:, 40 * 3:55 * 3],
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose_sample[:, 69:72] - tar_pose_sample[:, 69:72],
                reye_pose=tar_pose_sample[:, 72:75] - tar_pose_sample[:, 72:75],
            )


            facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :motion_length]
            facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :motion_length]
            face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
            l2 = self.reclatent_loss(facial_rec, facial_tar)
            self.l2_loss_all += l2.item() * n
            self.lvel_loss_all += face_vel_loss.item() * n

            joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127 * 3)[0, :n, :55 * 3]
            _ = self.l1_calculator.run(joints_rec)
            if self.alignmenter is not None:
                a_offset = int(self.align_mask * (self.cfg.DATASET.audio_sr / self.cfg.DATASET.pose_fps))
                onset_bt = self.alignmenter.load_audio(raw_audio[batch_index][:int(self.cfg.DATASET.audio_sr / self.cfg.DATASET.pose_fps * n)],
                                                       a_offset, len(raw_audio) - a_offset, True)
                beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n - self.align_mask, 30, True)
                # print(beat_vel)
                self.align_score_all += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n - 2 * self.align_mask))

            self.total_length += motion_length

        self.count += bs

