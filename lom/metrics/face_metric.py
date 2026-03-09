from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric
import pickle
import numpy as np

class FaceMetrics(Metric):
    def __init__(self,
                 cfg,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "Face metrics"
        self.region_path = cfg.METRIC.REGION_PATH
        with open(self.region_path, 'rb') as f:
            masks = pickle.load(f, encoding='latin1')
        self.template_path = cfg.METRIC.TEMPLATE_PATH
        self.template = torch.from_numpy(np.load(self.template_path)['mean_vertices']).float()

        # Common key names might be 'lips'/'mouth', 'upper_face' or need to combine 'eyes','brows','forehead' etc.
        self.lip_idx = masks['lips']
        self.upper_face_idx = np.concatenate([masks['eye_region'], masks['forehead']])
        
        # Add states for accumulating metrics
        self.add_state("LVE", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("FFD", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("MPVPE_FACE", default=torch.tensor(0.0), dist_reduce_fx="sum") 
        # self.add_state("MOD", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def compute_lve(self, vertices_pred, vertices_gt, lip_idx):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        L2_dis_mouth_max = torch.stack([torch.square(vertices_gt[:,v, :]-vertices_pred[:,v,:]) for v in lip_idx])
        L2_dis_mouth_max = torch.transpose(L2_dis_mouth_max, 0, 1)
        L2_dis_mouth_max = torch.sum(L2_dis_mouth_max, dim=2)
        L2_dis_mouth_max = torch.max(L2_dis_mouth_max, dim=1).values
        L2_dis_mouth_max = L2_dis_mouth_max.mean()
        return L2_dis_mouth_max

    def compute_ffd(self, vertices_pred, vertices_gt, upper_face_idx):
        """
        Facial Feature Distance
        """
        # Check if the template is on the same device as the vertices
        if self.template.device != vertices_pred.device:
            self.template = self.template.to(vertices_pred.device)

        # Calculate the motion of the predicted and ground truth vertices
        motion_pred = vertices_pred - self.template.reshape(1,-1,3)
        motion_gt = vertices_gt - self.template.reshape(1,-1,3)
        L2_dis_upper = torch.stack([torch.square(motion_gt[:,v, :]) for v in upper_face_idx])
        L2_dis_upper = torch.transpose(L2_dis_upper, 0, 1)
        L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
        L2_dis_upper = torch.std(L2_dis_upper, dim=0)
        gt_motion_std = torch.mean(L2_dis_upper)

        # Calculate the motion of the predicted vertices
        L2_dis_upper = torch.stack([torch.square(motion_pred[:,v, :]) for v in upper_face_idx])
        L2_dis_upper = torch.transpose(L2_dis_upper, 0, 1)
        L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
        L2_dis_upper = torch.std(L2_dis_upper, dim=0)
        pred_motion_std = torch.mean(L2_dis_upper)

        # Calculate the FFD
        FFD = gt_motion_std - pred_motion_std
        
        return FFD
    
    def compute_mpvpe(self, vertices_pred, vertices_gt):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        # Calculate the motion of the predicted and ground truth vertices
        mpvpe = torch.norm(vertices_pred- vertices_gt, p=2, dim=-1).mean(-1)
        return mpvpe
    
    @torch.no_grad()
    def update(self, rec_vertices: Tensor, tar_vertices: Tensor, lengths: Tensor = None):
        bs, n = rec_vertices.shape[0], rec_vertices.shape[1]
        self.count += bs
        # Calculate metrics
        for bs_idx in range(bs):
            length = lengths[bs_idx]
            self.LVE += self.compute_lve(rec_vertices[bs_idx, :length], tar_vertices[bs_idx, :length], self.lip_idx)
            self.FFD += self.compute_ffd(rec_vertices[bs_idx, :length], tar_vertices[bs_idx, :length], self.upper_face_idx)
            self.MPVPE_FACE += self.compute_mpvpe(1000*rec_vertices[bs_idx, :length], 1000*tar_vertices[bs_idx, :length]).mean()
            # self.MOD += self.compute_mod(rec_vertices[bs_idx, :length], tar_vertices[bs_idx, :length])

    def compute(self, sanity_flag=False):
        """Compute final metrics.
        Args:
            sanity_flag: Flag for sanity check, ignored in computation
        """
        metrics = {
            "LVE": self.LVE / self.count,
            "FFD": self.FFD / self.count,
            "MPVPE_FACE": self.MPVPE_FACE / self.count,
            # "MOD": self.MOD / self.count,
        }
        return metrics 