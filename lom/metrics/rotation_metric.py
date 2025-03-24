from typing import List
import torch
from torch import Tensor
from torchmetrics import Metric

class RotationMetrics(Metric):
    def __init__(self,
                 cfg,
                 dist_sync_on_step=True,
                 **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.cfg = cfg
        self.name = "Rotation metrics"

        # Add states for accumulating metrics
        self.add_state("mpjpe", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("mpvpe", default=torch.tensor(0.0), dist_reduce_fx="sum") 
        self.add_state("pampjpe", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("accel", default=torch.tensor(0.0), dist_reduce_fx="sum")        
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")


    def compute_mpjpe(self, preds, target, valid_mask=None, pck_joints=None, sample_wise=True):
        """
        Mean per-joint position error (i.e. mean Euclidean distance)
        often referred to as "Protocol #1" in many papers.
        """
        assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
        mpjpe = torch.norm(preds - target, p=2, dim=-1)  # BxJ

        if pck_joints is None:
            if sample_wise:
                mpjpe_seq = (
                    (mpjpe * valid_mask.float()).sum(-1) / valid_mask.float().sum(-1)
                    if valid_mask is not None
                    else mpjpe.mean(-1)
                )
            else:
                mpjpe_seq = mpjpe[valid_mask] if valid_mask is not None else mpjpe
            return mpjpe_seq
        else:
            mpjpe_pck_seq = mpjpe[:, pck_joints]
            return mpjpe_pck_seq


    def align_by_parts(self, joints, align_inds=None):
        if align_inds is None:
            return joints
        pelvis = joints[:, align_inds].mean(1)
        return joints - torch.unsqueeze(pelvis, dim=1)


    def calc_mpjpe(self, preds, target, align_inds=[0], sample_wise=True, trans=None):
        # Expects BxJx3
        valid_mask = target[:, :, 0] != -2.0
        # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
        if align_inds is not None:
            preds_aligned = self.align_by_parts(preds, align_inds=align_inds)
            if trans is not None:
                preds_aligned += trans
            target_aligned = self.align_by_parts(target, align_inds=align_inds)
        else:
            preds_aligned, target_aligned = preds, target
        mpjpe_each = self.compute_mpjpe(
            preds_aligned, target_aligned, valid_mask=valid_mask, sample_wise=sample_wise
        )
        return mpjpe_each


    def batch_compute_similarity_transform_torch(self, S1, S2):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.permute(0, 2, 1)
            S2 = S2.permute(0, 2, 1)
            transposed = True
        assert S2.shape[1] == S1.shape[1]

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = torch.sum(X1**2, dim=1).sum(dim=1)

        # 3. The outer product of X1 and X2.
        K = X1.bmm(X2.permute(0, 2, 1))

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, V = torch.svd(K)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
        Z = Z.repeat(U.shape[0], 1, 1)
        Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

        # Construct R.
        R = V.bmm(Z.bmm(U.permute(0, 2, 1)))

        # 5. Recover scale.
        scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

        # 6. Recover translation.
        t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

        # 7. Error:
        S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

        if transposed:
            S1_hat = S1_hat.permute(0, 2, 1)

        return S1_hat, (scale, R, t)

    def calc_pampjpe(self, preds, target, sample_wise=True, return_transform_mat=False):
        # Expects BxJx3
        target, preds = target.float(), preds.float()

        preds_tranformed, PA_transform = self.batch_compute_similarity_transform_torch(
            preds, target
        )
        pa_mpjpe_each = self.compute_mpjpe(preds_tranformed, target, sample_wise=sample_wise)

        if return_transform_mat:
            return pa_mpjpe_each, PA_transform
        else:
            return pa_mpjpe_each

    def calc_accel(self, preds, target):
        """
        Mean joint acceleration error
        often referred to as "Protocol #1" in many papers.
        """
        assert preds.shape == target.shape, print(preds.shape, target.shape)  # BxJx3
        assert preds.dim() == 3
        # Expects BxJx3
        # valid_mask = torch.BoolTensor(target[:, :, 0].shape)
        accel_gt = target[:-2] - 2 * target[1:-1] + target[2:]
        accel_pred = preds[:-2] - 2 * preds[1:-1] + preds[2:]
        normed = torch.linalg.norm(accel_pred - accel_gt, dim=-1)
        accel_seq = normed.mean(1)
        return accel_seq


    @torch.no_grad()
    def update(self, rec_joints: Tensor, tar_joints: Tensor, rec_vertices: Tensor, tar_vertices: Tensor, lengths: Tensor = None):

        bs, n = rec_joints.shape[0], rec_joints.shape[1]
        self.count += bs
        # Calculate metrics
        for bs_idx in range(bs):
            length = lengths[bs_idx]
            self.mpjpe += self.calc_mpjpe(rec_joints[bs_idx, :length] * 1000, tar_joints[bs_idx, :length] * 1000).mean()
            self.mpvpe += self.calc_mpjpe(rec_vertices[bs_idx, :length] * 1000, tar_vertices[bs_idx, :length] * 1000).mean()
            self.pampjpe += self.calc_pampjpe(rec_joints[bs_idx, :length] * 1000, tar_joints[bs_idx, :length] * 1000).mean()
            self.accel += self.calc_accel(rec_joints[bs_idx, :length] * 1000, tar_joints[bs_idx, :length] * 1000).mean()

    def compute(self, sanity_flag=False):
        """Compute final metrics.
        Args:
            sanity_flag: Flag for sanity check, ignored in computation
        """
        metrics = {
            "mpjpe": self.mpjpe / self.count,
            "mpvpe": self.mpvpe / self.count,
            "pampjpe": self.pampjpe / self.count,
            "accel": self.accel / self.count
        }
        return metrics 