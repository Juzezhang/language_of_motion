from typing import List
import torch
from torch import Tensor
import torch.nn as nn
from torchmetrics import Metric
from torchmetrics.functional import pairwise_euclidean_distance
from .utils import *
from lom.config import instantiate_from_config
from lom.archs.motion_representation import VAESKConv
from lom.utils import metric
from lom.optimizers.loss_factory import get_loss_func
from lom.metrics import data_tools
import torch.nn.functional as F
from os.path import join as pjoin


class CoSpeechMetrics(Metric):
    def __init__(self,
                 cfg,
                 dataname=None,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "matching, fid, and diversity scores"

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


        self.add_state("latent_out", default=[], dist_reduce_fx=None)
        self.add_state("latent_ori", default=[], dist_reduce_fx=None)
        self.add_state("motion_length", default=[], dist_reduce_fx=None)

        self.metrics = ["fid_score", "l2_loss", "lvel_loss", "align_score", "l1div"]

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_length", default=torch.tensor(0), dist_reduce_fx="sum")

        self.eval_model = VAESKConv(cfg.METRIC.CO_SPEECH.params)

        co_speech_checkpoint = torch.load(pjoin(cfg.DATASET.BEAT2.ROOT, cfg.METRIC.CO_SPEECH.e_path ), map_location="cpu", weights_only=False)

        new_state_dict = {} 
        for key, value in co_speech_checkpoint["model_state"].items():
            if key.startswith("module"):
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value

        self.eval_model.load_state_dict(new_state_dict)
        # other_tools.load_checkpoints(self.eval_model, pjoin(cfg.DATASET.BEAT2.ROOT, cfg.METRIC.CO_SPEECH.e_path ), cfg.METRIC.CO_SPEECH.e_name)
        self.eval_model.eval()


        self.avg_vel = np.load( pjoin( cfg.DATASET.BEAT2.ROOT, "weights/mean_vel_smplxflame_30.npy" ))
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
        vertices_rec: Tensor = None,
        vertices_rec_face: Tensor = None,
        vertices_tar_face: Tensor = None,
        raw_audio:Tensor = None,
        motion_lengths:Tensor = None):

        bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

        # self.motion_lengths.append(torch.tensor(motion_lengths).to(device))
        motion_lengths[motion_lengths > n] = n
        self.motion_lengths.append(motion_lengths)

        vert_rec_face = vertices_rec_face['vertices'].reshape(bs, n, 10475, 3)
        vert_tar_face = vertices_tar_face['vertices'].reshape(bs, n, 10475, 3)
        joints_rec_batch = vertices_rec["joints"].reshape(bs, n, 127, 3)

        for batch_index in range(bs):
            motion_length = motion_lengths[batch_index]
            if motion_length > n:
                motion_length = n
            remain = motion_length % self.cfg.vq.emage.params.vae_test_len

            self.latent_out.append(self.eval_model.map2latent(rec_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach())  # downsampling x16
            self.latent_ori.append(self.eval_model.map2latent(tar_pose[batch_index:batch_index+1, :motion_length - remain]).reshape(-1, 240).detach())


            facial_rec = vert_rec_face[batch_index].reshape(1, n, -1)[0, :motion_length].float()
            facial_tar = vert_tar_face[batch_index].reshape(1, n, -1)[0, :motion_length].float()
            face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
            l2 = self.reclatent_loss(facial_rec, facial_tar)

            self.l2_loss_all += l2.item() * motion_length
            self.lvel_loss_all += face_vel_loss.item() * motion_length

            joints_rec = joints_rec_batch[batch_index].float().detach().cpu().numpy().reshape(1, n, 127 * 3)[0, :motion_length, :55 * 3]
            _ = self.l1_calculator.run(joints_rec)
            if self.alignmenter is not None:
                a_offset = int(self.align_mask * (self.cfg.DATASET.audio_fps / self.cfg.DATASET.pose_fps))
                onset_bt = self.alignmenter.load_audio(raw_audio[batch_index][:int(self.cfg.DATASET.audio_fps / self.cfg.DATASET.pose_fps * motion_length)],
                                                       a_offset, len(raw_audio) - a_offset, True)
                beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, motion_length - self.align_mask, 30, True)
                # print(beat_vel)
                self.align_score_all += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (motion_length - 2 * self.align_mask))

            self.total_length += motion_length

        self.count += bs

