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


class AM2AMetrics_Exp(Metric):
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
        self.add_state("l1div", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rec_pose", default=[], dist_reduce_fx="cat")
        self.add_state("tar_pose", default=[], dist_reduce_fx="cat")
        self.add_state("motion_lengths", default=[], dist_reduce_fx="cat")

        # self.add_state("latent_out", default=torch.empty(0, 240), dist_reduce_fx="cat")
        # self.add_state("latent_ori", default=torch.empty(0, 240), dist_reduce_fx="cat")
        # self.add_state("motion_length", default=torch.empty(0, 1), dist_reduce_fx="cat")

        self.add_state("latent_out", default=[], dist_reduce_fx=None)
        self.add_state("latent_ori", default=[], dist_reduce_fx=None)
        self.add_state("motion_length", default=[], dist_reduce_fx=None)


        # self.latent_out = []
        # self.latent_ori = []


        self.metrics = ["fid_score", "l2_loss", "lvel_loss", "align_score", "l1div"]


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

    # @torch.no_grad()
    # def reset(self):
    #     # 重置 list 变量
    #     if isinstance(self.rec_pose, list):
    #         self.rec_pose.clear()
    #         self.tar_pose.clear()
    #         self.motion_lengths.clear()
    #     else:
    #         self.rec_pose = []
    #         self.tar_pose = []
    #         self.motion_lengths = []
    #
    #     # 重置其他 metric
    #     self.l2_loss = torch.tensor(0.0).to(self.device)
    #     self.lvel_loss = torch.tensor(0.0).to(self.device)
    #     self.fid_score = torch.tensor(0.0).to(self.device)
    #     self.align_score = torch.tensor(0.0).to(self.device)
    #
    #     self.l2_loss_all = torch.tensor(0.0).to(self.device)
    #     self.lvel_loss_all = torch.tensor(0.0).to(self.device)
    #     self.fid_score_all = torch.tensor(0.0).to(self.device)
    #     self.align_score_all = torch.tensor(0.0).to(self.device)
    #
    #     # Reset count and length trackers
    #     self.count = torch.tensor(0).to(self.device)
    #     self.total_length = torch.tensor(0).to(self.device)


    @torch.no_grad()
    def reset(self):
        # Reset tensor metrics to their initial values
        self.l2_loss = torch.tensor(0.0, device=self.device)
        self.l2_loss_all = torch.tensor(0.0, device=self.device)
        self.lvel_loss = torch.tensor(0.0, device=self.device)
        self.lvel_loss_all = torch.tensor(0.0, device=self.device)
        self.fid_score = torch.tensor(0.0, device=self.device)
        self.fid_score_all = torch.tensor(0.0, device=self.device)
        self.align_score = torch.tensor(0.0, device=self.device)
        self.align_score_all = torch.tensor(0.0, device=self.device)
        self.l1div = torch.tensor(0.0, device=self.device)

        # Reset count and total length
        self.count = torch.tensor(0, device=self.device)
        self.total_length = torch.tensor(0, device=self.device)

        # Clear list metrics
        if isinstance(self.rec_pose, list):
            self.rec_pose.clear()
        if isinstance(self.tar_pose, list):
            self.tar_pose.clear()
        if isinstance(self.motion_lengths, list):
            self.motion_lengths.clear()
        if isinstance(self.latent_out, list):
            self.latent_out.clear()
        if isinstance(self.latent_ori, list):
            self.latent_ori.clear()

        # # Reset additional states in l1_calculator and alignmenter if needed
        # self.l1_calculator.reset()
        # if self.alignmenter is not None:
        #     self.alignmenter.reset()





    @torch.no_grad()
    def compute(self, sanity_flag):

        count = self.count.item()
        total_length = self.total_length.item()
        # Init metrics dict
        metrics = {metric: getattr(self, metric) for metric in self.metrics}
        # Jump in sanity check stage
        if sanity_flag:
            return metrics

        device = self.l2_loss.device
        # latent_out_all = np.concatenate(self.latent_out, axis=0)
        # latent_ori_all = np.concatenate(self.latent_ori, axis=0)
        latent_out_all = torch.cat(self.latent_out, dim=0)
        latent_ori_all = torch.cat(self.latent_ori, dim=0)
        self.fid_score = torch.tensor(data_tools.FIDCalculator.frechet_distance(latent_out_all.cpu().numpy(), latent_ori_all.cpu().numpy())).to(device)
        self.l2_loss =self.l2_loss_all / total_length
        self.lvel_loss =self.lvel_loss_all / total_length
        self.align_score = self.align_score_all/(self.total_length-2* count*self.align_mask)
        self.l1div = torch.tensor(self.l1_calculator.avg()).to(device)

        metrics["fid_score"] = self.fid_score
        metrics["l2_loss"] = self.l2_loss
        metrics["lvel_loss"] = self.lvel_loss
        metrics["align_score"] = self.align_score
        metrics["l1div"] = self.l1div

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
        # self.smplx = self.smplx.to(torch.float)


        # select trainable joints
        smplx_2020 = smplx.create(
            self.cfg.DATASET.smpl_path,
            model_type='smplx',
            gender='NEUTRAL_2020',
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100,
            ext='npz',
            use_pca=False,
        ).eval()



        for batch_index in range(bs):
            motion_length = motion_lengths[batch_index]
            if motion_length > n:
                motion_length = n
            remain = motion_length % self.cfg.vq.emage.params.vae_test_len

            # self.latent_out.append(self.eval_model.map2latent(rec_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())  # downsampling x16
            # self.latent_ori.append(self.eval_model.map2latent(tar_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach().cpu().numpy())
            self.latent_out.append(self.eval_model.map2latent(rec_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach())  # downsampling x16
            self.latent_ori.append(self.eval_model.map2latent(tar_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach())
            rec_pose_sample = rc.rotation_6d_to_matrix(rec_pose[batch_index].reshape(n, j, 6))
            rec_pose_sample = rc.matrix_to_axis_angle(rec_pose_sample).reshape(n, j * 3)
            tar_pose_sample = rc.rotation_6d_to_matrix(tar_pose[batch_index].reshape(n, j, 6))
            tar_pose_sample = rc.matrix_to_axis_angle(tar_pose_sample).reshape(n, j * 3)


            # ### save npz for checking
            # rec_pose_np = rec_pose_sample.detach().cpu().numpy()
            # rec_trans_np = rec_trans[batch_index].detach().cpu().numpy().reshape( n, 3)
            # rec_exp_np = rec_exps[batch_index].detach().cpu().numpy().reshape( n, 100)
            # gt_pose_np = tar_pose_sample.detach().cpu().numpy()
            # gt_trans_np = tar_trans[batch_index].detach().cpu().numpy().reshape( n, 3)
            # gt_exp_np = tar_exps[batch_index].detach().cpu().numpy().reshape( n, 100)
            # np.savez('/data/code/exp_motion/motiongpt/results/rec.npz',
            #          betas=tar_beta[0, 0].detach().cpu().numpy(),
            #          poses=rec_pose_np - rec_pose_np,
            #          expressions=rec_exp_np,
            #          trans=rec_trans_np - rec_trans_np,
            #          model='smplx2020',
            #          gender='neutral',
            #          mocap_frame_rate=30,
            #          )
            # np.savez('/data/code/exp_motion/motiongpt/results/gt.npz',
            #          betas=tar_beta[0, 0].detach().cpu().numpy(),
            #          poses=gt_pose_np - gt_pose_np,
            #          expressions=gt_exp_np,
            #          trans=gt_trans_np - gt_trans_np,
            #          model='smplx2020',
            #          gender='neutral',
            #          mocap_frame_rate=30,
            #          )



            vertices_rec =smplx_2020(
                betas=tar_beta[batch_index].reshape(n, 300).cpu(),
                transl=rec_trans[batch_index].reshape(n, 3).cpu() - rec_trans[batch_index].reshape(n, 3).cpu(),
                expression=tar_exps[batch_index].reshape(n, 100).cpu() - tar_exps[batch_index].reshape(n, 100).cpu(),
                jaw_pose=rec_pose_sample[:, 66:69].cpu(),
                global_orient=rec_pose_sample[:, :3].cpu(),
                body_pose=rec_pose_sample[:, 3:21 * 3 + 3].cpu(),
                left_hand_pose=rec_pose_sample[:, 25 * 3:40 * 3].cpu(),
                right_hand_pose=rec_pose_sample[:, 40 * 3:55 * 3].cpu(),
                return_joints=True,
                leye_pose=rec_pose_sample[:, 69:72].cpu(),
                reye_pose=rec_pose_sample[:, 72:75].cpu(),
            )

            vertices_rec_face = smplx_2020(
                betas=tar_beta[batch_index].reshape(n, 300).cpu(),
                transl=rec_trans[batch_index].reshape(n, 3).cpu() - rec_trans[batch_index].reshape(n, 3).cpu(),
                expression=rec_exps[batch_index].reshape(n, 100).cpu(),
                jaw_pose=rec_pose_sample[:, 66:69].cpu(),
                global_orient=rec_pose_sample[:, :3].cpu() - rec_pose_sample[:, :3].cpu(),
                body_pose=rec_pose_sample[:, 3:21 * 3 + 3].cpu() - rec_pose_sample[:, 3:21 * 3 + 3].cpu(),
                left_hand_pose=rec_pose_sample[:, 25 * 3:40 * 3].cpu() - rec_pose_sample[:, 25 * 3:40 * 3].cpu(),
                right_hand_pose=rec_pose_sample[:, 40 * 3:55 * 3].cpu() - rec_pose_sample[:, 40 * 3:55 * 3].cpu(),
                return_verts=True,
                return_joints=True,
                leye_pose=rec_pose_sample[:, 69:72].cpu() - rec_pose_sample[:, 69:72].cpu(),
                reye_pose=rec_pose_sample[:, 72:75].cpu() - rec_pose_sample[:, 72:75].cpu(),
            )


            vertices_tar_face = smplx_2020(
                betas=tar_beta[batch_index].reshape(n, 300).cpu(),
                transl=tar_trans[batch_index].reshape(n, 3).cpu() - tar_trans[batch_index].reshape(n, 3).cpu(),
                expression=tar_exps[batch_index].reshape(n, 100).cpu(),
                jaw_pose=tar_pose_sample[:, 66:69].cpu(),
                global_orient=tar_pose_sample[:, :3].cpu() - tar_pose_sample[:, :3].cpu(),
                body_pose=tar_pose_sample[:, 3:21 * 3 + 3].cpu() - tar_pose_sample[:, 3:21 * 3 + 3].cpu(),
                left_hand_pose=tar_pose_sample[:, 25 * 3:40 * 3].cpu() - tar_pose_sample[:, 25 * 3:40 * 3].cpu(),
                right_hand_pose=tar_pose_sample[:, 40 * 3:55 * 3].cpu() - tar_pose_sample[:, 40 * 3:55 * 3].cpu(),
                return_verts=True,
                return_joints=True,
                leye_pose=tar_pose_sample[:, 69:72].cpu() - tar_pose_sample[:, 69:72].cpu(),
                reye_pose=tar_pose_sample[:, 72:75].cpu() - tar_pose_sample[:, 72:75].cpu(),
            )

            facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :motion_length].float()
            facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :motion_length].float()
            face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
            l2 = self.reclatent_loss(facial_rec, facial_tar)
            # self.l2_loss_all += l2.item() * n
            # self.lvel_loss_all += face_vel_loss.item() * n
            self.l2_loss_all += l2.item() * motion_length
            self.lvel_loss_all += face_vel_loss.item() * motion_length

            joints_rec = vertices_rec["joints"].float().detach().cpu().numpy().reshape(1, n, 127 * 3)[0, :motion_length, :55 * 3]
            _ = self.l1_calculator.run(joints_rec)
            if self.alignmenter is not None:
                a_offset = int(self.align_mask * (self.cfg.DATASET.audio_sr / self.cfg.DATASET.pose_fps))
                onset_bt = self.alignmenter.load_audio(raw_audio[batch_index][:int(self.cfg.DATASET.audio_sr / self.cfg.DATASET.pose_fps * motion_length)],
                                                       a_offset, len(raw_audio) - a_offset, True)
                beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, motion_length - self.align_mask, 30, True)
                # print(beat_vel)
                self.align_score_all += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (motion_length - 2 * self.align_mask))

            self.total_length += motion_length

        self.count += bs

